{ pkgs ? import <nixpkgs> {} }:
let
  pythonWithPkgs = (pkgs.python3.withPackages (p: with p; [
    einops
    jaxtyping
    jupyter
    jupytext
    matplotlib
    numpy
    omegaconf
    (open-clip-torch.overridePythonAttrs (o: {
      doCheck = false; # test are extremely slow
    }))
    pillow
    torch
    torchvision
    transformers
  ])).override (args: { ignoreCollisions = true; });
in
pkgs.mkShell {
  name = "dev-shell";
  nativeBuildInputs = with pkgs; [
    # Python
    pythonWithPkgs
    pyright
  ];
}
