{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/9607b9149c9d81fdf3dc4f3bcc278da146ffbd77.tar.gz") { } }:

pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (p: [
      p.numpy
      p.pandas
      p.matplotlib
      p.seaborn
      p.jupyter
      p.ipykernel
      p.scikit-learn
      p.tld
      p.pyqt6
      p.progress
    ]))
    pkgs.gnumake
  ];
}
