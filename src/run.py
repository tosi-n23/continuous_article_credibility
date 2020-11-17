import argparse
from api.app import create_app
from api.config import DevelopmentConfig, ProductionConfig


application = create_app(
    config_object=ProductionConfig)

def start_app(port):
    application.run(host='127.0.0.1', port=port, debug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--port', help="specify port",
                        type=int, default=8008)

    args = parser.parse_args()

    start_app(args.port)