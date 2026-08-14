import bioio_base.reader
import bioio_base.reader_metadata

###############################################################################


class ReaderMetadata(bioio_base.reader_metadata.ReaderMetadata):
    """
    Notes
    -----
    Defines metadata for the reader itself (not the image read), such as
    supported file extensions.
    """

    @staticmethod
    def get_supported_extensions() -> list[str]:
        return [".imzml"]

    @staticmethod
    def get_reader() -> type[bioio_base.reader.Reader]:  # ty: ignore[invalid-method-override]
        from .reader import Reader

        return Reader
