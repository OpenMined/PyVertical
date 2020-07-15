load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
   name = "org_openmined_psi",
   remote = "https://github.com/OpenMined/PSI",
   commit = "ca2866e3c70a018ed9a5a2e7af5bcbd31dc49df9",
   init_submodules = True,
)

load("@org_openmined_psi//private_set_intersection:preload.bzl", "psi_preload")

psi_preload()

load("@org_openmined_psi//private_set_intersection:deps.bzl", "psi_deps")

psi_deps()
