"""Generates layer selective directions."""

import functools

import torch
from pretorched import visualizers as vutils
from pretorched.gans import BigGAN, utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

res = 256
batch_size = 30
pretrained = "places365"
n_classes = {"places365": 365, "imagenet": 1000}.get(pretrained)
class_idx = 205  # lake with trees
G = None
g = None
batch_size = 1


def get_gs():
    """Load GAN generators from pretorched."""
    global G, g
    if G is None:
        G = BigGAN(resolution=res, pretrained=pretrained,
                   load_ema=True).to(device)
        g = functools.partial(G, embed=True)


def layer_regularizer_l1(G, zd, z, y, lay):
    """L1 regularizer on G activations.

    Args:
        G: GAN generator (from pretorched).
        zd: modified z vector.
        z: noise vector.
        y: class vector.
        lay: GAN layer.

    """
    zd_activations = G.forward(zd, y, embed=True, layer=lay)
    zd_activations_reshaped = zd_activations.view(zd_activations.size(0), -1)
    z_activations = G.forward(z, y, embed=True, layer=lay)
    z_activations_reshaped = z_activations.view(z_activations.size(0), -1)
    return (zd_activations_reshaped - z_activations_reshaped).abs().sum()


def layer_regularizer_L2(G, zd, z, y, lay):
    """L2 regularizer on G activations.

    Args:
        G: GAN generator (from pretorched).
        zd: modified z vector.
        z: noise vector.
        y: class vector.
        lay: GAN layer.

    """
    zd_activations = G.forward(zd, y, embed=True, layer=lay)
    zd_activations_reshaped = zd_activations.view(zd_activations.size(0), -1)
    z_activations = G.forward(z, y, embed=True, layer=lay)
    z_activations_reshaped = z_activations.view(z_activations.size(0), -1)
    a = torch.abs(zd_activations_reshaped - z_activations_reshaped)
    return a.pow(2).mean()


def choose_orthog_z(G, mat):
    """Choose directions orthogonal to subspace.

    Args:
        G: GAN generator (from pretorched).
        mat: subspace basis vectors (d in previous layers).

    """
    if mat.nelement() == 0:  # if empty, start with random d
        d, _ = utils.prepare_z_y(1, G.dim_z,
                                 n_classes, device=device, z_var=0.5)
        basis_mat = []
        return d, basis_mat  # generate any starting d
    basis_mat = torch.qr(mat.t())[0].t()
    some_z = torch.randn(119).cuda()
    z_in_subspace = torch.mm(basis_mat.t(),
                             torch.mm(basis_mat, some_z[:, None]))[:, 0]
    d = some_z - z_in_subspace  # make starting d orthogonal to subspace
    return d, basis_mat


def optimize_lsds(
    z,
    num_dirs_per_layer,
    num_samples,
    start_layer,
    end_layer,
    visualize,
    learning_rate,
    new_class,
    savedirs,
    savepath,
):
    """Generate layer selective directions.

    Args:
        z: starting noise vector.
        num_dirs__per_layer: number of LSDs per layer.
        num_samples: steps in optimization looop.
        start_layer: layer of G (closest to output) where
            LSD generation will start.
        end_layer: layer of G (closest to latent space)
            where LSD generation will finish.
        visualize: bool determining whether to generate
            G(z+LSD) per LSD.
        learning_rate: lr for optimization.
        new_class: image class in which LSDs are generated.
        savedirs: bool determining whether LSDs are saved.
        savepath: directory where LSDs are saved.


    """
    y = new_class * torch.ones(1, device=device).long()
    get_gs()
    G_z = utils.elastic_gan(g, z, y)
    print("Original z")
    vutils.visualize_samples(G_z)
    dnorm_mat = torch.tensor(()).cuda()  # init as empty

    for layer in range(start_layer, end_layer - 1, -1):

        dnorm_mat_layer = []  # collect LSDs per layer here

        for j in range(num_dirs_per_layer):
            d, basis_mat = choose_orthog_z(dnorm_mat)
            d = torch.nn.Parameter(d, requires_grad=True)
            optimizer = torch.optim.Adam([d], lr=learning_rate)

            with torch.enable_grad():  # optimization loop
                for step_num in range(num_samples):
                    optimizer.zero_grad()  # zero gradients
                    if layer == start_layer:  # no need to enforce orthog.
                        dnorm = d / torch.norm(d)  # normalize first
                    else:  # enforce orthogonality constraint
                        d_in_subspace = torch.mm(
                            basis_mat.t(), torch.mm(basis_mat, d[:, None])
                        )[:, 0]
                        d_orthog = d - d_in_subspace
                        dnorm = d_orthog / torch.norm(d_orthog)
                    zd = z + 10 * dnorm  # forward pass, alpha = 10
                    loss = 0.02 * layer_regularizer_L2(G, zd, z, y, layer)
                    loss.backward  # optimize loss
                    optimizer.step()

            if layer == start_layer:
                dnorm = d / torch.norm(d)
            else:
                d_in_subspace = torch.mm(
                    basis_mat.t(), torch.mm(basis_mat, d[:, None])
                )[:, 0]
                d_orthog = d - d_in_subspace
                dnorm = d_orthog / torch.norm(d_orthog)

            dnorm_mat_layer.append(dnorm)

            if visualize:  # careful about memory if you create a lot
                z_mod = z + 10 * dnorm
                G_zmod = utils.elastic_gan(g, z_mod, y)
                print("G(z+d) for d ", j, "in layer ", layer)
                vutils.visualize_samples(G_zmod)

        dnorm_mat_layer = torch.stack(dnorm_mat_layer)
        dnorm_mat = torch.cat((dnorm_mat, dnorm_mat_layer), 0)
        dnorm_mat = torch.squeeze(dnorm_mat)

    if savedirs:
        path = f"{savepath}/LSDs_L{start_layer}_{end_layer}"
        torch.save(dnorm_mat, path)
