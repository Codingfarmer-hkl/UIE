import json

class RecordSchema:
    def __init__(self, type_list, role_list, type_role_dict):
        self.type_list = type_list
        self.role_list = role_list
        self.type_role_dict = type_role_dict

    def __repr__(self) -> str:
        return f"Type: {self.type_list}\n" \
            f"Role: {self.role_list}\n" \
            f"Map: {self.type_role_dict}"

    @staticmethod
    def get_empty_schema():
        return RecordSchema(type_list=list(), role_list=list(), type_role_dict=dict())

    @staticmethod
    def read_from_file(filename):
        lines = open(filename, "r", encoding="utf-8").readlines()
        type_list = json.loads(lines[0])
        role_list = json.loads(lines[1])
        type_role_dict = json.loads(lines[2])
        return RecordSchema(type_list, role_list, type_role_dict)

    def write_to_file(self, filename):
        with open(filename, 'w') as output:
            output.write(json.dumps(self.type_list) + '\n')
            output.write(json.dumps(self.role_list) + '\n')
            output.write(json.dumps(self.type_role_dict) + '\n')

