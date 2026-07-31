# Local embedding models

TinyContext downloads the selected ONNX embedding bundle here when this
checkout directory is selected explicitly through configuration.

Native pip and uvx installs use the platform-specific TinyContext data
directory. Docker uses `/data/models`. Model files are intentionally ignored.
