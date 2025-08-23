from fast_pytorch_kmeans import KMeans, MultiKMeans
import psutil, os
import faiss
import torch
import numpy as np
import pickle
from safetensors.torch import load_file, save_file
import argparse
from functools import partial


print("FAISS version:", faiss.__version__)
print("FAISS built with GPU support:", hasattr(faiss, "StandardGpuResources"))


"""
In this file, we run K-Means using a batched GPU implementation and then save computed centroids
in order to later use them in our compression pipeline
This file implements the calibration step of TemporalKV, ChannelKV and KVQuant  
"""


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-8
    return (x - mean) / std


def kmeans_reconstruction_error_batched(
    x: torch.Tensor,
    fisher_dir: str,
    k: int = 4096,
    W=None,
    provided_kmeans=None,
    fisher_mse=False,
    test_mse=True,
    device="cuda",
) -> float:

    """
    Compression function for ChannelKV without normalization
    """

    x_np = x.to(device).detach().float()
    if W is not None:
        W = W.to(device).detach().float()
    bsz = x_np.shape[0]
    d = x_np.shape[2]
    train_ids = int(x_np.shape[1] * 0.75)
    if provided_kmeans is None:
        kmeans = MultiKMeans(n_clusters=k, mode="euclidean", init_method="kmeans++")
        if W is not None:
            kmeans.fit_predict(
                x_np[:, :train_ids], weights=W[:, :train_ids], fisher_dir=fisher_dir
            )
        else:
            kmeans.fit_predict(x_np[:, :train_ids])
    else:
        kmeans = provided_kmeans

    # Assign to nearest centroids
    if test_mse:
        x_test = x_np[:, train_ids:]
        if W is not None:
            W_test = W[:, train_ids:]
    else:
        x_test = x_np[:, :train_ids]
        if W is not None:
            W_test = W[:, :train_ids]
    I = kmeans.predict(x_test)
    x_recon = kmeans.centroids[torch.arange(bsz).unsqueeze(-1), I.squeeze()]

    if fisher_mse:
        mse_fisher = (W_test * ((x_test - x_recon) ** 2)).mean()
        return float(mse_fisher), kmeans
    else:
        mse = ((x_test - x_recon) ** 2).mean()
        return float(mse), kmeans


def kmeans_reconstruction_error_norm_batched(
    x: torch.Tensor,
    fisher_dir: str,
    k: int = 4096,
    provided_kmeans=None,
    weighted=False,
    remove_outliers=0,
    W=None,
    fisher_mse=False,
    test_mse=True,
    device="cuda",
    var="not_specified",
    outlier_mask=None,
    provided_means=None,
    provided_stds=None,
    quantile_norm=False,
    provided_upper_quantile=None,
    provided_lower_quantile=None,
) -> float:
    """
    Compression function for TemporalKV and KVQuant with normalization
    """
    x_np = (
        x.to(device).detach().float()
    )  # (bsz,seqlen,size_centroids,size_centroids) ou (bsz,seqlen,size_centroids,1)
    if fisher_dir == "uniform":
        print("Attention uniform kv quant")
    if x_np.shape[-1] == 1:
        assert var != "not_specified"
    if var == "k":
        x2 = x.to(device)
        x3 = x2.transpose(0, 1).flatten(1, 2).squeeze(-1)  # (seqlen,bsz*size_centroids)
        outlier_mask, lower_quantile, upper_quantile = get_outlier_mask(
            x3, remove_outliers
        )
        outlier_mask = outlier_mask.view(x2.shape)
        if quantile_norm:
            shape = (x_np.shape[0], 1, x_np.shape[2], x_np.shape[3])
            upper_quantile = upper_quantile.view(shape)
            lower_quantile = lower_quantile.view(shape)
            scale = (upper_quantile - lower_quantile) / 2
    elif var == "v":
        if remove_outliers > 0:
            assert outlier_mask is not None
        if quantile_norm:
            provided_upper_quantile = provided_upper_quantile.view(
                1, x_np.shape[1], 1, 1
            )
            provided_lower_quantile = provided_lower_quantile.view(
                1, x_np.shape[1], 1, 1
            )
            scale = (provided_upper_quantile - provided_lower_quantile) / 2
    else:
        assert var == "not_specified"
    if remove_outliers > 0:
        outlier_mask = outlier_mask.to(device)
        print(f"mask mean : {outlier_mask.float().mean()}")
    if remove_outliers == 0:
        outlier_mask = None
    if x_np.shape[-1] == 1:
        assert var in ["k", "v"]
        kvq = True
    else:
        kvq = False

    if W is not None:
        W = W.to(device)
    bsz = x_np.shape[0]
    d = x_np.shape[3]
    train_ids = int(x_np.shape[1] * 0.75)
    if var == "v":
        assert x_np.shape[-1] == 1
        means = provided_means.to(device)
        stds = provided_stds.to(device)  # (1,seqlen,1,1)
    else:
        means = x_np.mean(dim=(1, 3), keepdim=True)
        stds = (x_np - means).std(dim=(1, 3), keepdim=True)  # (bsz,1,size_centroids,1)
    if not quantile_norm:
        scale = stds
    scale = scale.cuda()
    if provided_kmeans is None:
        if var == "v":
            print("x_np,means,scale")
            print(x_np.shape, means.shape, scale.shape)
            x_train = (
                (x_np[:, :train_ids] - means[:, :train_ids]) / scale[:, :train_ids]
            ).reshape(
                bsz, -1, d
            )  # (bsz,train_ids*size_centroids,size_centroids)
        else:
            x_train = ((x_np[:, :train_ids] - means) / scale).reshape(
                bsz, -1, d
            )  # (bsz,train_ids*size_centroids,size_centroids)
        kmeans = MultiKMeans(n_clusters=k, mode="euclidean", init_method="kmeans++")
        if weighted:
            if fisher_dir in ["fisher", "fisher_reloaded"]:
                if not kvq:
                    weights = (
                        scale.repeat(1, train_ids, 1, 1) ** 2
                    )  # (bsz,train_ids,size_centroids,1)
                    assert len(weights.flatten()) == x_train.shape[0] * x_train.shape[1]
                    assert W is not None
                    if W is not None:
                        weights = (
                            weights * W[:, :train_ids]
                        )  # (bsz,train_ids,size_centroids,size_centroids)
                else:
                    assert W is not None
                    if var == "v":
                        weights = (
                            scale[:, :train_ids].repeat(1, 1, size_centroids, 1) ** 2
                        )  # (bsz,train_ids,size_centroids)
                    else:
                        weights = (
                            scale.repeat(1, train_ids, 1, 1) ** 2
                        )  # (bsz,train_ids,size_centroids,1)
                    print(weights.shape)
                    print(W.shape)
                    print(train_ids)
                    weights = weights * W[:, :train_ids]
                    if remove_outliers > 0:
                        weights = weights * (1 - outlier_mask[:, :train_ids].float())
            else:
                assert fisher_dir == "uniform"
                assert W is None
                weights = (
                    scale.squeeze(-1).repeat(1, train_ids, 1) ** 2
                )  # (bsz,train_ids,size_centroids)
            weights = weights.to(device)
            kmeans.fit_predict(
                x_train, weights=weights.reshape(bsz, -1, d), fisher_dir=fisher_dir
            )
            centroids = kmeans.centroids
        else:
            kmeans.fit_predict(x_train)
            centroids = kmeans.centroids
    else:
        kmeans = provided_kmeans
        centroids = kmeans.centroids

    # Assign to nearest centroids
    if test_mse:
        x_test = x_np[:, train_ids:].contiguous()
        if W is not None:
            W_test = W[:, train_ids:]
        if remove_outliers > 0:
            test_mask = 1 - outlier_mask[:, train_ids:].float()
        if var == "v":
            means_test = means[:, train_ids:]
            scale_test = scale[:, train_ids:]
    else:
        x_test = x_np[:, :train_ids].contiguous()
        if W is not None:
            W_test = W[:, :train_ids]
        if remove_outliers > 0:
            test_mask = 1 - outlier_mask[:, :train_ids].float()
        if var == "v":
            means_test = means[:, :train_ids]
            scale_test = scale[:, :train_ids]
    if var == "k" or var == "not_specified":
        means_test, scale_test = means, scale
    print(f"remove outliers : {remove_outliers}")
    x_test_collapsed = ((x_test - means_test) / scale_test).view(bsz, -1, d)
    I = kmeans.predict(x_test_collapsed)
    x_recon = kmeans.centroids[torch.arange(bsz).unsqueeze(-1), I]
    x_recon = x_recon.view((bsz, x_test.shape[1], x_test.shape[2], x_test.shape[3]))
    if fisher_mse:
        assert W is not None
        diff = W_test * ((x_test - x_recon * scale_test - means_test) ** 2)
        if remove_outliers > 0:
            diff = diff * test_mask
        mse = diff.flatten().mean()
    else:
        diff = (x_test - x_recon * scale_test - means_test) ** 2
        if remove_outliers > 0:
            diff = diff * test_mask
        mse = diff.flatten().mean()
    return float(mse), kmeans


### The following compression functions are alternative and not used in our main pipeline


def outlier_residual_quantizer(x_test, kmeans, k2, percentile, W2=None, weighted=True):

    """
    Test time compression function using Residual Quantization on outliers
    """

    assert len(x_test.shape) == 4
    x_np = x_test.cpu().detach().float()
    n_heads = x_np.shape[0]
    d = x_np.shape[3]
    means = x_np.mean(dim=(1, 3)).reshape(n_heads, 1, -1, 1)
    stds = ((x_np - means).std(dim=(1, 3))).reshape(n_heads, 1, -1, 1)
    x_final = (x_np - means) / stds
    Lresiduals = []
    W = []
    for i in range(0, n_heads):
        stdsi = stds[i].repeat(x_final.shape[1], 1, 1).flatten()
        x_finali = x_final[i].view(-1, d)
        D, I = kmeans.index.search(x_finali, 1)
        x_reconi = kmeans.centroids[I.squeeze()]
        residualsi = x_finali - x_reconi
        normsresidualsi = torch.norm(residualsi, dim=1)
        threshold = torch.quantile(normsresidualsi, 1 - percentile)
        top_indices = (normsresidualsi >= threshold).nonzero(as_tuple=True)[0]
        Lresiduals.append(residualsi[top_indices])
        if W2 is not None:
            W2i = W2[i].flatten()
            W.append(W2i[top_indices] * stdsi[top_indices] ** 2)
        else:
            W.append(stdsi[top_indices] ** 2)
    residuals_train = torch.cat(Lresiduals)
    assert residuals_train.shape[0] > 3000
    kmeans2 = faiss.Kmeans(
        d=d, k=k2, niter=50, verbose=False, gpu=torch.cuda.is_available()
    )
    if weighted:
        kmeans2.train(residuals_train, weights=torch.cat(W).flatten().numpy())
    else:
        kmeans2.train(residuals_train)
    return kmeans2


def train_selective_residual_quantizer(x, k1, k2, percentile, W=None, weighted=True):

    """
    Training time compression function using Residual Quantization
    """

    main_ids = int(x.shape[1] * 0.66)
    x_main = x[:, :main_ids]
    x_residual = x[:, main_ids:]

    # Train main quantizer
    x_np = x_main.cpu().detach().float()
    assert len(x_np.shape) == 4
    n_heads = x_np.shape[0]
    d = x_np.shape[3]
    means = x_np.mean(dim=(1, 3)).reshape(n_heads, 1, -1, 1)
    stds = ((x_np - means).std(dim=(1, 3))).reshape(n_heads, 1, -1, 1)

    x_train = ((x_np - means) / stds).view(-1, d).numpy()
    kmeans1 = faiss.Kmeans(
        d=d, k=k1, niter=50, verbose=False, gpu=torch.cuda.is_available()
    )
    if weighted:
        vars = stds.squeeze(-1).repeat(1, x_np.shape[1], 1) ** 2
        weights = vars / torch.sum(vars, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)
        if W is not None:
            weights = weights * W[:, :main_ids]
        kmeans1.train(x_train, weights=weights.flatten().numpy())
    else:
        kmeans1.train(x_train)

    # Train residual quantizer
    if W is not None:
        kmeans2 = outlier_residual_quantizer(
            x_residual, kmeans1, k2, percentile, W2=W[:, main_ids:]
        )
    else:
        kmeans2 = outlier_residual_quantizer(
            x_residual, kmeans1, k2, percentile, W2=None
        )

    return kmeans1, kmeans2


def train_coupled_codebooks(x, k1, k2, pred_matrix, weighted=True):

    """
    Coupled Codebooks training-time function
    In this experimental function (not part of the thesis), we experimented how a linear prediction matrix could allow to reduce the error
    on half of the chunks by using their preceding chunk to predict them
    """

    main_ids = int(x.shape[1] * 0.66)

    # Reshape and normalize
    x_np = x.cpu().detach().float()
    assert len(x_np.shape) == 4
    n_heads = x_np.shape[0]
    d = x_np.shape[3]
    means = x_np.mean(dim=(1, 3)).reshape(n_heads, 1, -1, 1)
    stds = ((x_np - means).std(dim=(1, 3))).reshape(n_heads, 1, -1, 1)
    x_train = (x_np - means) / stds

    x_base = x_train[:, :main_ids].reshape(-1, d).numpy()
    x_2 = x_train[:, main_ids:].reshape(-1, d).numpy()

    kmeans1 = faiss.Kmeans(
        d=d // 2, k=k1, niter=50, verbose=False, gpu=torch.cuda.is_available()
    )
    print("train kmeans 1 mean abs :" + str(np.mean(np.abs(x_base[:, : d // 2]))))
    kmeans1.train(x_base[:, : d // 2])

    D2, I2 = kmeans1.index.search(x_2[:, : d // 2], 1)
    recon_x_21 = kmeans1.centroids[I2.squeeze()]

    x_22 = x_2[:, d // 2 :]

    # Using previous prediction to approximate half of the chunks
    x_22 = x_22 - recon_x_21 @ pred_matrix

    kmeans2 = faiss.Kmeans(
        d=d // 2, k=k2, niter=50, verbose=False, gpu=torch.cuda.is_available()
    )
    kmeans2.train(x_22)
    print("train kmeans2 mean abs :" + str(np.mean(np.abs(x_22))))

    return kmeans1, kmeans2


def evaluate_coupled_codebooks(x, kmeans1, kmeans2, pred_matrix):

    """
    Coupled Codebooks test-time function
    In this experimental function (not part of the thesis), we experimented how a linear prediction matrix could allow to reduce the error
    on half of the chunks by using their preceding chunk to predict them
    """

    # normalization step
    x_np = x.cpu().detach().float()
    d = x_np.shape[2]
    means = x_np.mean(dim=(0, 2)).reshape(1, -1, 1)
    stds = (x_np - means).std(dim=(0, 2)).reshape(1, -1, 1)
    x_normalized = ((x_np - means) / stds).reshape(-1, d)

    # First quantization
    D1, I1 = kmeans1.index.search(x_normalized[:, : d // 2], 1)
    x_recon1 = kmeans1.centroids[I1.squeeze()]

    x_2 = x_normalized[:, d // 2 :] - x_recon1 @ pred_matrix

    D2, I2 = kmeans2.index.search(x_2, 1)
    x_recon2 = kmeans2.centroids[I2.squeeze()] + x_recon1 @ pred_matrix

    # Final reconstruction
    x_recon = torch.Tensor(np.concatenate([x_recon1, x_recon2], axis=1))
    x_finalrecon_reshaped = x_recon.view((x_np.shape[0], x_np.shape[1], x_np.shape[2]))
    final_diff = (x_np - x_finalrecon_reshaped * stds - means) ** 2
    mse = final_diff.mean()

    x_2bis = x_normalized[:, d // 2 :]
    D2bis, I2bis = kmeans1.index.search(x_2bis, 1)
    x_recon2bis = kmeans1.centroids[I2bis.squeeze()]
    x_reconbis = torch.Tensor(np.concatenate([x_recon1, x_recon2bis], axis=1))
    x_finalrecon_reshapedbis = x_reconbis.view(
        (x_np.shape[0], x_np.shape[1], x_np.shape[2])
    )
    final_diffbis = (x_np - x_finalrecon_reshapedbis * stds - means) ** 2
    msebis = final_diffbis.mean()

    return float(mse), float(msebis)


def evaluate_selective_residual_quantizer(
    x, kmeans1, kmeans2, percentile, fisher_mse=False, W=None
):

    """
    Test time compression function using Residual Quantization
    """

    # normalization step
    x_np = x.cpu().detach().float()
    d = x_np.shape[2]
    means = x_np.mean(dim=(0, 2)).reshape(1, -1, 1)
    stds = (x_np - means).std(dim=(0, 2)).reshape(1, -1, 1)
    x_normalized = ((x_np - means) / stds).reshape(-1, d)

    # First quantization
    D1, I1 = kmeans1.index.search(x_normalized, 1)
    x_recon = torch.Tensor(kmeans1.centroids[I1.squeeze()])
    x_recon_reshaped = x_recon.view((x_np.shape[0], x_np.shape[1], x_np.shape[2]))
    if fisher_mse and W is not None:
        diff = W.unsqueeze(-1) * ((x_np - x_recon_reshaped * stds - means) ** 2)
    else:
        diff = (x_np - x_recon_reshaped * stds - means) ** 2
    normsresiduals = diff.sum(dim=-1).flatten()
    mse1 = torch.mean(diff)

    # Selection of bad residuals
    threshold = torch.quantile(normsresiduals, 1 - percentile)
    top_indices = (normsresiduals >= threshold).nonzero(as_tuple=True)[0]
    bad_residuals = x_normalized[top_indices] - x_recon[top_indices]

    # Quantization of bad residuals
    D2, I2 = kmeans2.index.search(bad_residuals, 1)
    recon_bad_residuals = kmeans2.centroids[I2.squeeze()]

    # Final reconstruction
    x_recon[top_indices] = x_recon[top_indices] + recon_bad_residuals
    x_finalrecon_reshaped = x_recon.view((x_np.shape[0], x_np.shape[1], x_np.shape[2]))
    if fisher_mse and W is not None:
        final_diff = W.unsqueeze(-1) * (
            (x_np - x_finalrecon_reshaped * stds - means) ** 2
        )
    else:
        final_diff = (x_np - x_finalrecon_reshaped * stds - means) ** 2
    final_diff = final_diff.reshape(-1, d)
    final_diff[top_indices, :] = 0
    mse2 = final_diff.mean()

    return float(mse1), float(mse2)



""" Data manipulation functions """


def preprocess_gtensor(tensor, n_heads=8, dim_keys=128):
    n_samples = tensor.shape[0]
    seqlen = tensor.shape[1]
    n_heads = tensor.shape[2] // dim_keys
    out = (
        tensor.reshape((n_samples, seqlen, n_heads, dim_keys))
        .transpose(2, 0)
        .transpose(1, 2)[:, :, 8:, :]
    )  # (n_heads,n_samples,seqlen,dim_keys) (1,self.num_heads,1,head_dim)
    return out


def preprocess_data(data, dim_var):
    acts_k = preprocess_gtensor(data["acts_k"], dim_keys=dim_var)
    acts_v = preprocess_gtensor(data["acts_v"], dim_keys=dim_var)
    grads2_k = preprocess_gtensor(data["grads_k"], dim_keys=dim_var)
    grads2_v = preprocess_gtensor(data["grads_v"], dim_keys=dim_var)
    return acts_k, acts_v, grads2_k, grads2_v


def get_chan_temp_kvquant(X, size_centroids, i):
    Xi = X[:, :, i * size_centroids : i * size_centroids + size_centroids]
    X_chan = Xi.contiguous().reshape(-1, size_centroids)
    X_temp = (
        Xi.contiguous()
        .reshape(
            (Xi.shape[0], Xi.shape[1] // size_centroids, size_centroids, size_centroids)
        )
        .transpose(2, 3)
        .reshape(-1, size_centroids, size_centroids)
    )
    X_kvquant = Xi.reshape(Xi.shape[0] * Xi.shape[1], Xi.shape[2]).unsqueeze(-1)
    return X_chan, X_temp, X_kvquant


def get_chan_temp_kvquant_tot(X, size_centroids):
    Xi = (
        X.reshape(X.shape[0], X.shape[1], X.shape[2] // size_centroids, size_centroids)
        .transpose(0, 2)
        .transpose(1, 2)
        .contiguous()
    )
    X_chan = Xi.contiguous().reshape(X.shape[2] // size_centroids, -1, size_centroids)
    X_temp = (
        Xi.contiguous()
        .reshape(
            (
                Xi.shape[0],
                Xi.shape[1],
                Xi.shape[2] // size_centroids,
                size_centroids,
                size_centroids,
            )
        )
        .transpose(3, 4)
        .reshape(X.shape[2] // size_centroids, -1, size_centroids, size_centroids)
    )
    X_kvquant = Xi.reshape(
        Xi.shape[0], Xi.shape[1] * Xi.shape[2], Xi.shape[3]
    ).unsqueeze(-1)
    return X_chan, X_temp, X_kvquant


def get_outlier_mask(x, th, dim=2):
    # x has shape (batch,size_centroids or dim_keys)
    ### Inspired from KVQuant's outlier removal functions
    lower_quantile = torch.quantile(x, th, dim=0, keepdim=True)
    upper_quantile = torch.quantile(x, 1 - th, dim=0, keepdim=True)
    under = x <= lower_quantile
    upper = x >= upper_quantile

    outlier_mask = torch.logical_or(under, upper)
    return outlier_mask, lower_quantile, upper_quantile


def get_per_token_stats(acts_v, t):
    # acts_v has shape (n_heads, n_samples, seqlen, dim_keys)
    # acts has shape (n_heads, dim_keys, n_samples, seqlen)
    acts = acts_v
    n_heads, n_samples, seqlen, dim_keys = acts.shape
    acts = acts.permute(0, 3, 1, 2).reshape(
        -1, n_samples * seqlen
    )  # (batch,n_samples*seqlen)
    outlier_mask, lower_quantile, upper_quantile = get_outlier_mask(acts, t)
    upper_quantile = upper_quantile.view(1, n_samples * seqlen, 1, 1)
    lower_quantile = lower_quantile.view(1, n_samples * seqlen, 1, 1)
    outlier_mask = outlier_mask.view(n_heads, dim_keys, n_samples, seqlen).permute(
        0, 2, 3, 1
    )

    means = acts_v.mean(dim=(0, 3), keepdim=False).view(1, -1, 1, 1)
    stds = acts_v.std(dim=(0, 3), keepdim=False).view(1, -1, 1, 1)
    return outlier_mask, means, stds, lower_quantile, upper_quantile


#####################
#####################
##### Main Loop #####
#####################
#####################


if __name__ == "__main__":
    print("here at beginning of main")
    parser = argparse.ArgumentParser()

    parser.add_argument("--nbits", type=int, default=8, help="Number of bits")
    parser.add_argument(
        "--fisher", type=str, default="no", help="Use Fisher information"
    )
    parser.add_argument(
        "--size_centroids", type=int, default=4, help="Size of centroids"
    )
    parser.add_argument(
        "--output_dir", type=str, default="centroids/", help="Output Directory"
    )
    parser.add_argument(
        "--start_layer", type=int, default=0, help="First layer to calculate"
    )
    parser.add_argument(
        "--end_layer", type=int, default=32, help="First layer to calculate"
    )
    parser.add_argument(
        "--nb_examples",
        type=int,
        default=22,
        help="Number of examples used to calibrate data",
    )
    parser.add_argument("--norm", type=str, default="norm", help="baselines")
    parser.add_argument("--mean_only", type=str, default="no", help="baselines")
    parser.add_argument("--model_name", type=str, default="no", help="baselines")

    args = parser.parse_args()

    nbits = args.nbits
    fisher = args.fisher
    size_centroids = args.size_centroids
    output_dir = args.output_dir
    start_layer = args.start_layer
    end_layer = args.end_layer
    model_name = args.model_name  # "LLaMA-2-7B-32K"
    fisher_dir = args.fisher
    nb_examples = args.nb_examples

    fisher = fisher_dir in ["fisher", "fisher_reloaded", "fisher_kvquant"]
    if not fisher:
        fisher_mse = False
    else:
        fisher_mse = True
    if not fisher_dir in ["fisher", "fisher_reloaded", "fisher_kvquant", "uniform"]:
        raise ValueError("fisher not conventional, got " + args.fisher)

    if args.norm == "norm":
        is_norm = True
    elif args.norm == "baselines":
        is_norm = False
    else:
        raise ValueError(
            "args.norm not conventional : either norm or baselines, nothing else"
        )
    if args.mean_only == "yes":
        mean_only = True
    elif args.mean_only == "no":
        mean_only = False
    else:
        raise ValueError(
            "args.mean_only not conventional : either yes or no, nothing else"
        )

    print("Mean only is : " + str(mean_only))
    print("Quant type is : " + args.norm)
    print("Fisher is : " + fisher_dir)
    print("Layer " + str(start_layer) + " to " + str(end_layer))
    print("Size centroids : " + str(size_centroids))
    print("Nbits : " + str(nbits))

    n_centroids = int(pow(2, nbits))
    k1 = n_centroids
    k2 = 256
    high_precision = [1] * 128
    centroids_temp = [high_precision[i] * n_centroids for i in range(0, 128)]
    thresh = 0.99
    t = (1 - thresh) / 2
    provided_kmeans = None
    if model_name == "LLaMA-2-7B-32K":
        n_heads = 32
    elif (
        model_name == "Llama-3.1-8B"
        or model_name == "Llama-3.2-3B"
        or model_name == "Llama-3.2-1B"
    ):
        n_heads = 8
    else:
        raise ValueError("model not known, please add it")
    percentile = 0.01
    percentile_kvquant = 0.01
    res_display = 7
    coeff = 1.0
    if model_name in ["LLaMA-2-7B-32K", "Llama-3.1-8B", "Llama-3.2-3B"]:
        dim_var = 128
    elif model_name == "Llama-3.2-1B":
        dim_var = 64
    else:
        raise ValueError("model not implemented")
    type_quant = "norm"
    test_mse = True

    if fisher_mse:
        assert fisher_dir in ["fisher_kvquant", "fisher", "fisher_reloaded"]

    if not mean_only:
        """
        Activations and gradients collection pipeline
        """
        if is_norm:
            MSES_norm = []
        else:
            MSES_chan = []
            MSES_kvquant = []
            MSES_kvquant1p = []
        for layer in range(start_layer, end_layer):  ###ATTENTION REPLACE
            if is_norm:
                mses_norm = []
            else:
                mses_chan = []
                mses_kvquant = []
                mses_kvquant1p = []
            print("_________________")
            print("_________________")
            print("____LAYER" + str(layer) + "____")
            print("_________________")
            print("_________________")
            if nb_examples <= 22:
                data = load_file(
                    "gtensors/"
                    + model_name
                    + "/"
                    + "layer"
                    + str(layer)
                    + "_nb"
                    + str(nb_examples)
                    + ".safetensors"
                )
            else:
                data = None
                for i in range(0, nb_examples // 8):
                    if data is None:
                        assert i == 0
                        data = load_file(
                            "gtensors/"
                            + model_name
                            + "/"
                            + "layer"
                            + str(layer)
                            + "_nb"
                            + str(nb_examples)
                            + "split"
                            + str(i)
                            + ".safetensors"
                        )
                    else:
                        added_data = load_file(
                            "gtensors/"
                            + model_name
                            + "/"
                            + "layer"
                            + str(layer)
                            + "_nb"
                            + str(nb_examples)
                            + "split"
                            + str(i)
                            + ".safetensors"
                        )
                        for obj_type in ["acts_k", "acts_v", "grads_k", "grads_v"]:
                            data[obj_type] = torch.cat(
                                [data[obj_type], added_data[obj_type]], dim=0
                            )
                print("Split data successfully loaded")
            acts_k, acts_v, grads2_k, grads2_v = preprocess_data(data, dim_var)
            (
                outlier_mask_v,
                provided_means,
                provided_stds,
                provided_upper_quantile,
                provided_lower_quantile,
            ) = get_per_token_stats(acts_v, t)
            # acts and outlier mask have shape (n_heads,n_samples,seqlen,dim_keys)
            print("________grads2_k________")
            print(f"min : {grads2_k.min()}")
            print(f"max : {grads2_k.max()}")

            if not is_norm:
                acts_temp_k = (
                    acts_k.contiguous()
                    .reshape(
                        (
                            acts_k.shape[0],
                            acts_k.shape[1],
                            acts_k.shape[2] // size_centroids,
                            size_centroids,
                            dim_var,
                        )
                    )
                    .transpose(3, 4)
                    .reshape(n_heads, -1, dim_var, size_centroids)
                )
                acts_temp_v = (
                    acts_v.contiguous()
                    .reshape(
                        (
                            acts_v.shape[0],
                            acts_v.shape[1],
                            acts_v.shape[2] // size_centroids,
                            size_centroids,
                            dim_var,
                        )
                    )
                    .transpose(3, 4)
                    .reshape(n_heads, -1, dim_var, size_centroids)
                )
                if fisher:
                    weights_temp_k = (
                        grads2_k.contiguous()
                        .reshape(
                            (
                                grads2_k.shape[0],
                                grads2_k.shape[1],
                                grads2_k.shape[2] // size_centroids,
                                size_centroids,
                                dim_var,
                            )
                        )
                        .transpose(3, 4)
                        .reshape(n_heads, -1, dim_var, size_centroids)
                        .sum(dim=-1)
                    )
                    weights_temp_v = (
                        grads2_v.contiguous()
                        .reshape(
                            (
                                grads2_v.shape[0],
                                grads2_v.shape[1],
                                grads2_v.shape[2] // size_centroids,
                                size_centroids,
                                dim_var,
                            )
                        )
                        .transpose(3, 4)
                        .reshape(n_heads, -1, dim_var, size_centroids)
                        .sum(dim=-1)
                    )

                # The commented code here is old code when we tested residual quantization and centroid sharing across whole heads, which we leave for future work
                # provided_kmeans1k,provided_kmeans2k=train_selective_residual_quantizer(acts_temp_k.contiguous().float(),k1,k2,percentile,W=None if not fisher else weights_temp_k.contiguous().float(),weighted=True)
                # provided_kmeans1v,provided_kmeans2v=train_selective_residual_quantizer(acts_temp_v.contiguous().float(),k1,k2,percentile,W=None if not fisher else weights_temp_v.contiguous().float(),weighted=True)

                # C_common1k,C_common2k=torch.Tensor(provided_kmeans1k.centroids),torch.Tensor(provided_kmeans2k.centroids)
                # C_common1v,C_common2v=torch.Tensor(provided_kmeans1v.centroids),torch.Tensor(provided_kmeans2v.centroids)
                # C_common={"k1":C_common1k,"k2":C_common2k,"v1":C_common1v,"v2":C_common2v}
                C_chan = {"k": [], "v": []}
                C_kvquant = {"k": [], "v": []}
                C_kvquant1p = {"k": [], "v": []}
            else:
                C_norm = {"k": [], "v": []}

            for var in ["v", "k"]:
                print("n_heads : " + str(n_heads))
                for head_num in range(0, n_heads):  ####ATTENTION REPLACE
                    if is_norm:
                        sum_norm = 0
                    else:
                        sum_chan = 0
                        sum_norm2 = 0
                        sum_resout = 0
                        sum_kvquant = 0
                        sum_kvquant1p = 0
                    print("head_num : " + str(head_num))
                    if var == "k":
                        Amask = None
                        A = acts_k[head_num]
                        if fisher:
                            G2A = grads2_k[head_num]
                    elif var == "v":
                        A = acts_v[head_num]
                        assert outlier_mask_v.shape[-1] == dim_var
                        Amask = (
                            outlier_mask_v[head_num]
                            .flatten(0, 1)
                            .reshape(-1, dim_var // size_centroids, size_centroids)
                            .transpose(0, 1)
                            .contiguous()
                            .unsqueeze(-1)
                        )  # (bsz,seqlen,size_centroids,1)
                        if fisher:
                            G2A = grads2_v[head_num]
                    print("activations collected")
                    C_chani = []
                    C_kvquanti = []
                    C_normi = []
                    ## debut batched ##
                    A_chan, A_temp, A_kvquant = get_chan_temp_kvquant_tot(
                        A, size_centroids
                    )
                    if fisher:
                        G2A_chan, G2A_temp, G2A_kvquant = get_chan_temp_kvquant_tot(
                            G2A, size_centroids
                        )
                        W_chan, W_temp, W_kvquant = (
                            G2A_chan,
                            G2A_temp,
                            G2A_kvquant * size_centroids,
                        )
                    if not is_norm:
                        print("mse chan")
                        mse_chan, kmeans_chan = kmeans_reconstruction_error_batched(
                            A_chan.float(),
                            W=None if not fisher else W_chan.float(),
                            k=n_centroids,
                            fisher_mse=fisher_mse,
                            test_mse=test_mse,
                            fisher_dir=fisher_dir,
                        )
                        C_chan_head = kmeans_chan.centroids
                        print("mse kvquant")
                        # outlier mask has shape (n_samples,seqlen,dim_keys)
                        # A kvquant has shape (bsz,n_samples*seqlen,size_centroids,1)
                        # provided_quantile has shape (1,n_samples*seqlen,1,1)
                        (
                            mse_kvquant1p,
                            kmeans_kvquant1p,
                        ) = kmeans_reconstruction_error_norm_batched(
                            A_kvquant.float(),
                            W=None if not fisher else W_kvquant.float(),
                            k=int(pow(n_centroids, 1 / size_centroids)),
                            remove_outliers=t,
                            fisher_mse=fisher_mse,
                            weighted=True,
                            test_mse=test_mse,
                            fisher_dir=fisher_dir,
                            provided_means=provided_means,
                            provided_stds=provided_stds,
                            var=var,
                            outlier_mask=Amask,
                            quantile_norm=True,
                            provided_upper_quantile=provided_upper_quantile,
                            provided_lower_quantile=provided_lower_quantile,
                        )
                        (
                            mse_kvquant,
                            kmeans_kvquant,
                        ) = kmeans_reconstruction_error_norm_batched(
                            A_kvquant.float(),
                            W=None if not fisher else W_kvquant.float(),
                            k=int(pow(n_centroids, 1 / size_centroids)),
                            remove_outliers=0,
                            fisher_mse=fisher_mse,
                            weighted=True,
                            test_mse=test_mse,
                            fisher_dir=fisher_dir,
                            provided_means=provided_means,
                            provided_stds=provided_stds,
                            var=var,
                        )

                        C_kvquant_head = kmeans_kvquant.centroids
                        C_kvquant1p_head = kmeans_kvquant1p.centroids
                    else:
                        print("mse norm")
                        (
                            mse_norm,
                            kmeans_norm,
                        ) = kmeans_reconstruction_error_norm_batched(
                            A_temp.float(),
                            W=None if not fisher else W_temp.float(),
                            k=centroids_temp[0],
                            fisher_mse=fisher_mse,
                            weighted=True,
                            test_mse=test_mse,
                            fisher_dir=fisher_dir,
                        )
                        C_norm_head = kmeans_norm.centroids
                    if is_norm:
                        C_norm[var].append(C_norm_head)
                    else:
                        C_chan[var].append(C_chan_head)
                        C_kvquant[var].append(C_kvquant_head)
                        C_kvquant1p[var].append(C_kvquant1p_head)

                    print(".............................")
                    print(".............................")
                    print("VAR : " + str(var))
                    print("HEAD N° " + str(head_num) + " :")
                    if is_norm:
                        print("Norm : " + str(mse_norm))
                        mses_norm.append(mse_norm)
                    else:
                        print("Chan : " + str(mse_chan))
                        mses_chan.append(mse_chan)
                        # print("Common Centroids : "+str(sum_norm2/(dim_var//size_centroids)))
                        # print("Common Centroids1% : "+str(sum_resout/(dim_var//size_centroids)))
                        print("KvQuant : " + str(mse_kvquant))
                        print("KvQuant1p : " + str(mse_kvquant1p))
                        mses_kvquant.append(mse_kvquant)
                        mses_kvquant1p.append(mse_kvquant1p)
                    print(".............................")
                    print(".............................")

                if is_norm:
                    C_norm[var] = torch.stack(C_norm[var])
                else:
                    C_chan[var] = torch.stack(C_chan[var])
                    C_kvquant[var] = torch.stack(C_kvquant[var])
                    C_kvquant1p[var] = torch.stack(C_kvquant1p[var])
            # Save files
            subdir = fisher_dir + "/"
            file_string = (
                subdir
                + "layer"
                + str(layer)
                + "size"
                + str(size_centroids)
                + "bit"
                + str(nbits)
                + "nbexamples"
                + str(nb_examples)
                + model_name
                + ".safetensors"
            )
            if is_norm:
                save_file(C_norm, output_dir + "norm/" + file_string)
            else:
                # save_file(C_common,output_dir+"common/"+file_string)
                save_file(C_chan, output_dir + "chan/" + file_string)
                save_file(C_kvquant, output_dir + "kvquant/" + file_string)
                print(output_dir + "kvquant1p/" + file_string)
                save_file(C_kvquant1p, output_dir + "kvquant1p/" + file_string)
            if is_norm:
                MSES_norm.append(mses_norm)
            else:
                MSES_chan.append(mses_chan)
                MSES_kvquant.append(mses_kvquant)
                MSES_kvquant1p.append(mses_kvquant1p)
        ## save matrixes ##
        file_string = (
            "size"
            + str(size_centroids)
            + "bit"
            + str(nbits)
            + "nbexamples"
            + str(nb_examples)
            + model_name
        )
        output_dir = "mses/"
        if is_norm:
            np.save(
                output_dir + "norm" + "_" + fisher_dir + "_" + file_string + ".npy",
                np.array(MSES_norm),
            )
        else:
            np.save(
                output_dir + "chan" + "_" + fisher_dir + "_" + file_string + ".npy",
                np.array(MSES_chan),
            )
            np.save(
                output_dir + "kvquant" + "_" + fisher_dir + "_" + file_string + ".npy",
                np.array(MSES_kvquant),
            )
            np.save(
                output_dir
                + "kvquant1p"
                + "_"
                + fisher_dir
                + "_"
                + file_string
                + ".npy",
                np.array(MSES_kvquant),
            )
    else:
        """General statistics computation pipeline"""
        for layer in range(start_layer, end_layer):
            print("_________________")
            print("_________________")
            print("____LAYER" + str(layer) + "____")
            print("_________________")
            print("_________________")
            if nb_examples <= 22:
                data = load_file(
                    "gtensors/"
                    + model_name
                    + "/"
                    + "layer"
                    + str(layer)
                    + "_nb"
                    + str(nb_examples)
                    + ".safetensors"
                )
            else:
                data = None
                for i in range(0, nb_examples // 8):
                    if data is None:
                        assert i == 0
                        data = load_file(
                            "gtensors/"
                            + model_name
                            + "/"
                            + "layer"
                            + str(layer)
                            + "_nb"
                            + str(nb_examples)
                            + "split"
                            + str(i)
                            + ".safetensors"
                        )
                    else:
                        added_data = load_file(
                            "gtensors/"
                            + model_name
                            + "/"
                            + "layer"
                            + str(layer)
                            + "_nb"
                            + str(nb_examples)
                            + "split"
                            + str(i)
                            + ".safetensors"
                        )
                        for obj_type in ["acts_k", "acts_v", "grads_k", "grads_v"]:
                            data[obj_type] = torch.cat(
                                [data[obj_type], added_data[obj_type]], dim=0
                            )
            acts_k, acts_v, grads2_k, grads2_v = preprocess_data(data, dim_var)
            # acts has shape (n_heads,n_samples,seqlen,dim_keys)
            acts_k_reshaped = (
                acts_k.transpose(0, 2).flatten(0, 1).flatten(1, 2)
            )  # (n_samples*len,n_heads*dim_keys)
            lower_quantile = torch.quantile(
                acts_k_reshaped, t, dim=0, keepdim=True
            )  # (1,n_heads*dim_keys)
            upper_quantile = torch.quantile(
                acts_k_reshaped, 1 - t, dim=0, keepdim=True
            )  # (1,n_heads*dim_keys)
            key_quantiles = torch.cat((lower_quantile, upper_quantile), dim=0)

            # we want shape (1,self.num_heads,1,head_dim)
            acts_k = acts_k.transpose(0, 1)
            acts_v = acts_v.transpose(0, 1)
            # acts has shape (n_samples,n_heads,seqlen,dim_keys)
            means_k = acts_k.mean(dim=(0, 2), keepdim=True)
            means_v = acts_v.mean(dim=(0, 2), keepdim=True)
            stds_k = (acts_k - means_k).std(dim=(0, 2), keepdim=True)
            stds_v = (acts_v - means_v).std(dim=(0, 2), keepdim=True)
            # Save files
            file_string = (
                "nbexamples"
                + str(nb_examples)
                + "layer"
                + str(layer)
                + "means_stds_"
                + model_name
                + ".safetensors"
            )

            dic_stats = {}
            dic_stats["mean_k"] = means_k
            dic_stats["mean_v"] = means_v
            dic_stats["stds_k"] = stds_k
            dic_stats["stds_v"] = stds_v
            dic_stats["key_quantiles"] = key_quantiles
            save_file(dic_stats, output_dir + "stats/" + file_string)
