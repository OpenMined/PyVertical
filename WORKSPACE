load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
   name = "org_openmined_psi",
   remote = "https://github.com/OpenMined/PSI",
   branch = "fix_c++_version",
   init_submodules = True,
)

load("@org_openmined_psi//private_set_intersection:preload.bzl", "psi_preload")

psi_preload()

load("@org_openmined_psi//private_set_intersection:deps.bzl", "psi_deps")

psi_deps()
