import telegram
import os
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
            try:
                self.bot.send_message(chat_id=channel, text=self.training_name +"@ " +text)
            except:
                print("Telegram: send text failed")

    def send_file(self, path):
        for channel in self.channels:
            try:
                self.bot.send_document(chat_id=channel, document=open(path, "rb"))
            except:
                print("Telegram: send text failed")

    def send_image(self, path):
        for channel in self.channels:
            try:
                self.bot.send_photo(chat_id=channel, photo=open(path, "rb"))
            except:
                print("Telegram: send text failed")

    def emit(self, record):
        log_entry = self.format(record)
        self.send_text(log_entry)