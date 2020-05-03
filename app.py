import json

from utils.flask_utils import (
    Cronometer,
    read_image_from_request,
    process_args,
    set_up_tensorflow,
)

from flask import Flask, Response, request
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app)


@app.route("/heartbeat")
def hello():
    return Response(
        response=json.dumps({"alive": True}),
        status=200,
        mimetype="application/json",
    )


@app.route("/api/caption", methods=["POST"])
def caption():
    cronometer = Cronometer()
    print("New request:")

    try:
        image = read_image_from_request(request.data)
        print("\tRead image in %f seconds." % cronometer.record_time())

        model_output = predict_caption_fn(image)
        print("\tRan model in %f seconds." % cronometer.record_time())

        caption = postprocess_caption_fn(model_output)
        print("\tPost-processed output in %f seconds." % cronometer.record_time())

        print("\tTotal time spent was %f seconds." % cronometer.record_time(from_start=True))

        response = json.dumps({"success": True, "caption": str(caption)})
        return Response(
            response=response,
            status=200,
            mimetype="application/json",
        )
    except Exception as e:
        # TODO: Better error management.
        error_response = json.dumps({
            "success": False,
            "error_message": "An error occurred while processing your request. Please try again later.",
        })
        return Response(
            response=error_response,
            status=500,
            mimetype="application/json",
        )


if __name__ == "__main__":
    args = process_args()
    predict_caption_fn, postprocess_caption_fn = set_up_tensorflow(
        args.frozen_model_filename,
        args.config_filename,
        args.gpu_memory,
    )
    app.run()
