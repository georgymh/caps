import os
import json
import logging

from utils.flask_utils import (
    Cronometer,
    read_image_from_request,
    process_args,
    set_up_tensorflow,
)

from flask import Flask, Response, request
from flask_cors import CORS

logging.basicConfig(
   level=logging.DEBUG,
   format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
   datefmt='%Y-%m-%d %H:%M:%S',
   handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

def create_app():
    app = Flask(__name__)
    cors = CORS(app)


    @app.route("/heartbeat")
    def heartbeat():
        logger.info("Heartbeat request successful!")
        return Response(
            response=json.dumps({"alive": True}),
            status=200,
            mimetype="application/json",
        )


    @app.route("/api/caption", methods=["POST"])
    def caption():
        logger.info("New caption request:")

        try:
            cronometer = Cronometer()
            image = read_image_from_request(request.data)
            logger.info("\tRead image in %f seconds." % cronometer.record_time())

            model_output = predict_caption_fn(image)
            logger.info("\tRan model in %f seconds." % cronometer.record_time())

            caption = postprocess_caption_fn(model_output)
            logger.info("\tPost-processed output in %f seconds." % cronometer.record_time())

            logger.info("\tTotal time spent was %f seconds." % cronometer.record_time(from_start=True))

            response = json.dumps({"success": True, "caption": str(caption)})
            return Response(
                response=response,
                status=200,
                mimetype="application/json",
            )
        except Exception as e:
            # TODO: Better error management.
            logger.error("\tFound error during request!")
            error_response = json.dumps({
                "success": False,
                "error_message": "An error occurred while processing your request. Please try again later.",
            })
            return Response(
                response=error_response,
                status=500,
                mimetype="application/json",
            )


    return app


if __name__ == "__main__":
    logger.info("Reading args to app...")
    args = process_args()

    logger.info("Setting up tensorflow model...")
    predict_caption_fn, postprocess_caption_fn = set_up_tensorflow(
        args.frozen_model_filename,
        args.config_filename,
        args.gpu_memory,
    )

    logger.info("Creating app...")
    app = create_app()

    logger.info("Starting app...")
    app.run(host='0.0.0.0', debug=True)
