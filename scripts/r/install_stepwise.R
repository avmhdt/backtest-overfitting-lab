# Install StepwiseTest into the user library, from Posit's Ubuntu 22.04 binaries.
library <- Sys.getenv("R_LIBS_USER")
dir.create(library, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(library, .libPaths()))
options(
  HTTPUserAgent = sprintf(
    "R/%s R (%s)",
    getRversion(),
    paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])
  )
)
install.packages(
  "StepwiseTest",
  lib = library,
  repos = "https://packagemanager.posit.co/cran/__linux__/jammy/latest"
)
library(StepwiseTest)
cat("StepwiseTest", as.character(packageVersion("StepwiseTest")), "\n")
