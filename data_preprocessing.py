# -*- coding:utf-8 -*-
from collections import Counter
import os
import json
from typing import Dict, List
from tqdm import tqdm
import argparse
from data_preprocessing.text2spotasoc import Text2SpotAsoc
from data_preprocessing.structure_marker import BaseStructureMarker
from data_preprocessing.dataset import Dataset
from data_preprocessing.ie_format import Sentence


def convert_graph(args: argparse, datasets: Dict[str, List[Sentence]],
                  language: str = "en", label_mapper: Dict = None):
    convertor = Text2SpotAsoc(structure_maker=BaseStructureMarker(), language=language, label_mapper=label_mapper)
    counter = Counter()
    os.makedirs(args.save_path, exist_ok=True)
    schema_counter = {
        "entity": list(),
        "relation": list(),
        "event": list(),
    }
    for data_type, instance_list in datasets.items():
        with open(os.path.join(args.save_path, f"{data_type}.json"), "w", encoding="utf-8") as output:
            for instance in tqdm(instance_list):
                counter.update([f"{data_type} sent"])
                converted_graph = convertor.annonote_graph(
                    tokens=instance.tokens,
                    entities=instance.entities,
                    relations=instance.relations,
                    events=instance.events,
                )
                src, tgt, spot_labels, asoc_labels = converted_graph[:4]
                spot_asoc = converted_graph[4]

                schema_counter["entity"] += instance.entities
                schema_counter["relation"] += instance.relations
                schema_counter["event"] += instance.events

                output.write(
                    "%s\n"
                    % json.dumps(
                        {
                            "text": src,
                            "tokens": instance.tokens,
                            "record": tgt,
                            "entity": [
                                entity.to_offset(label_mapper)
                                for entity in instance.entities
                            ],
                            "relation": [
                                relation.to_offset(
                                    ent_label_mapper=label_mapper,
                                    rel_label_mapper=label_mapper,
                                )
                                for relation in instance.relations
                            ],
                            "event": [
                                event.to_offset(evt_label_mapper=label_mapper)
                                for event in instance.events
                            ],
                            "spot": list(spot_labels),
                            "asoc": list(asoc_labels),
                            "spot_asoc": spot_asoc,
                        },
                        ensure_ascii=False,
                    )
                )
    convertor.output_schema(os.path.join(args.save_path, "record.schema"))
    convertor.get_entity_schema(schema_counter["entity"]).write_to_file(
        os.path.join(args.save_path, f"entity.schema")
    )
    convertor.get_relation_schema(schema_counter["relation"]).write_to_file(
        os.path.join(args.save_path, f"relation.schema")
    )
    convertor.get_event_schema(schema_counter["event"]).write_to_file(
        os.path.join(args.save_path, f"event.schema")
    )

    with open(f"{args.save_path}/{args.save_statistics_data_filename}", "w", encoding="utf-8") as f:
        json.dump(dict(counter), f, ensure_ascii=False, indent=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default="data_preprocessing/data_config/own.yaml")
    parser.add_argument("-save_path", type=str, default="data/train_data")
    parser.add_argument("-save_statistics_data_filename", type=str, default='statistics_train_data.json')
    args = parser.parse_args()
    dataset = Dataset.load_yaml_file(args.config)
    datasets = dataset.load_dataset()
    label_mapper = dataset.mapper
    convert_graph(args, datasets=datasets, language=dataset.language, label_mapper=label_mapper)


if __name__ == "__main__":
    main()
