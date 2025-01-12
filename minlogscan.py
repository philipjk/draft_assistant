import json
import os
import logging
from utils import load_arena_id_map


class LogParser():
    def __init__(self, log_path, expansion_code="FDN"):
        self.log_path = log_path
        self.log_position = 0
        self.arena_file_size = 0
        self.draft_start = False
        self.pool = []
        self.pack = []
        self.pick_number = 0
        self.arena_id_map = load_arena_id_map(expansion_code=expansion_code)
        self.last_pick = None

    def read_log(self):
        while True:
            self.search_draft_start()
            if self.draft_start:
                self.search_draft_data()


    def search_draft_data(self):
        if len(self.pool) == 0:
            # no pack data for pack 1 pick 1
            self.search_p1p1()
        else:
            self.search_pack()
            self.search_pick()
            
    
    def arenaid2names(self, card_list):
        card_names = []
        for card_id in card_list:
            card_name = self.arena_id_map.get(card_id, "Unknown Card")
            card_names.append(card_name)
        return card_names

    def search_pick(self):
        flag = False
        with open(self.log_path, 'r',
                    encoding='utf-8',
                    errors='replace') as self.log_file:
            self.log_file.seek(self.log_position)
            while True:
                line = self.log_file.readline()
                if not line:
                    flag = False
                    break
                if 'DraftMakePick' in line:
                    start_offset = line.find('{"id"')
                    payload_data = json.loads(line[start_offset:])
                    request = json.loads(payload_data['request'])
                    self.pool.append(request['GrpId'])
                    self.last_pick = request['GrpId']
                    self.pick_number += 1
                    logging.info(f"Pack {request['Pack']} Pick {request['Pick']} made.")
                    flag = True
                    break
            self.log_position = self.log_file.tell()
        return flag

    def search_pack(self):
        flag = False
        with open(self.log_path, 'r',
                    encoding='utf-8',
                    errors='replace') as self.log_file:
            self.log_file.seek(self.log_position)
            while True:
                line = self.log_file.readline()
                if not line:
                    flag = False
                    break
                if 'PackCards' in line:
                    start_offset = line.find('{"draftId"')
                    payload_data = json.loads(line[start_offset:])
                    self.pack = [int(card) for card in payload_data['PackCards'].split(',')]
                    logging.info(f"Pack data found.")
                    flag = True
                    break
            self.log_position = self.log_file.tell()
        return flag

    def search_p1p1(self):
        flag = False
        with open(self.log_path, 'r',
                    encoding='utf-8',
                    errors='replace') as self.log_file:
            self.log_file.seek(self.log_position)
            while True:
                line = self.log_file.readline()
                if not line:
                    flag = False
                if 'MakePick' in line:
                    logging.info("Pack 1 Pick 1. No pack data.")
                    start_offset = line.find('{"id"')
                    payload_data = json.loads(line[start_offset:])
                    request = json.loads(payload_data['request'])
                    self.pool.append(request['GrpId'])
                    self.pick_number += 1
                    flag = True
                    break
            self.log_position = self.log_file.tell()
        return flag

    def search_draft_start(self):
        # TODO: add a try/except for when the log file does not exist yet
        file_size = os.path.getsize(self.log_path)
        if file_size < self.arena_file_size:
            # if you start the logger before arena
            logging.info("Found a new log file.")
            self.log_position = 0
            self.arena_file_size = 0
            self.draft_start = False
            self.pool = [] 
        elif file_size > self.arena_file_size:
            self.arena_file_size = file_size

        if not self.draft_start:
            with open(self.log_path, 'r',
                        encoding='utf-8',
                        errors='replace') as self.log_file:
                self.log_file.seek(self.log_position)
                while True:
                    line = self.log_file.readline()
                    if not line:
                        break
                    if "PremierDraft" in line:
                        self.draft_start = True
                        logging.info("Draft started.")
                        break
                self.log_position = self.log_file.tell()
    