import telegram
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import logging
import yaml

class TelegramLog(logging.Handler):

    def __init__(self, conf_path):
        super(TelegramLog, self).__init__()
        self.conf = yaml.load(open(conf_path))
        self.token = self.conf["token"]
        self.training_name = self.conf["name"]
        self.channels = self.conf["channels"]
        self.bot = telegram.Bot(token=self.token)

    def send_text(self, text):
        for channel in self.channels:
            self.bot.send_message(chat_id=channel, text=self.training_name +"@ " +text)

    def send_file(self, path):
        for channel in self.channels:
            self.bot.send_document(chat_id=channel, document=open(path, "rb"))

    def send_image(self, path):
        for channel in self.channels:
            self.bot.send_photo(chat_id=channel, photo=open(path, "rb"))

    def emit(self, record):
        log_entry = self.format(record)
        self.send_text(log_entry)