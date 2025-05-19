class DataNotFoundError(Exception):
    """Exception raised when the requested data is not found in the data source."""
    def __init__(self, message="Requested data not found in the file system"):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message

class UnknownError(Exception):
    """Exception raised for generic errors during data retrieval or processing."""

    def __init__(self, message="Uknown error occurred while retrieving data"):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message
