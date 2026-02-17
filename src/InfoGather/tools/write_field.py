from src.InfoGather.tools.base import InfoBookTool

SUCCESS_MESSAGE = "Successfully wrote '{value}' to field '{field_name}'"
ERROR_FIELD_NOT_EXIST = "Error: Field '{field_name}' does not exist"


class WriteFieldTool(InfoBookTool):
    @property
    def name(self) -> str:
        return "write_field"

    @property
    def description(self) -> str:
        return "Write a value to a field in the info book. Use this to save information you've gathered from the user."

    async def execute(self, field_name: str, value: str) -> str:
        """
        Write a value to a field.

        Args:
            field_name: The name of the field to write to.
            value: The value to write to the field.
        """
        error = self.info_book.set_field_value(field_name, value)
        if error is None:
            return SUCCESS_MESSAGE.format(value=value, field_name=field_name)

        field = self.info_book.get_field(field_name)
        if not field:
            return ERROR_FIELD_NOT_EXIST.format(field_name=field_name)

        return f"Error: {error}"
