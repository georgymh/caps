# Caps Server

https://getcaps.apps

## Testing

1. To run **all unit and integration tests** run:
```
pytest
```

2. To quickly test the **ML model execution logic** run:
```
python execute_model.py -m models/gramcapsv0_optimized.pb --input_image assets/test_image.jpg --auto_infer_tensor_names --show_caption
```

3. To test the **Caps Server** locally run:
```
./start.sh
```
You'll then be able to hit the API at `localhost:8080`.


4. To test the prepping of the (docker) **Environment with the Caps Server** running run:
```
docker run --rm -it -p 8080:8080 $(docker build -q .)
```
Similarly, you'll be able to hit the API at `localhost:8080`.
