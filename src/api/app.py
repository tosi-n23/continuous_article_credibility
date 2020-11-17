from flask import Flask

from .config import get_logger


_logger = get_logger(logger_name=__name__)


def create_app(*, config_object) -> Flask:
    """Create a flask app instance."""

    flask_app = Flask('ac_api')
    flask_app.config.from_object(config_object)

    # import blueprints
    from .controller import article_credibility_app
    flask_app.register_blueprint(article_credibility_app)
    _logger.debug('Application instance created')

    return flask_app