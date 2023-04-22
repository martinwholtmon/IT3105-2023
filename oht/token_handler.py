from decouple import config


def get_token():
    return config("OHT_TOKEN")
