import warnings


def configure_runtime_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+",
        category=Warning,
        module=r"urllib3",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument\.",
        category=UserWarning,
        module=r"torch\.functional",
    )
