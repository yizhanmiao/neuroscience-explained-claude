export JULIA_PKG_USE_CLI_GIT := "true"
export GIT_LFS_SKIP_SMUDGE := "1"
JULIA := "julia +1.12"

pluto:
    {{ JULIA }} -e "using Pluto; Pluto.run()"

run NOTEBOOK:
    {{ JULIA }} -e 'using Pluto; Pluto.run(notebook="{{ NOTEBOOK }}")'
