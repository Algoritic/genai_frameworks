import json
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
import jsonschema


class JSONSchemaParser(BaseOutputParser[str]):

    def parse(self, schema: dict) -> str:
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.SchemaError as e:
            raise OutputParserException(f"Invalid JSON schema: {e}")
        return schema

    @property
    def _type(self) -> str:
        return "json_schema_parser"
