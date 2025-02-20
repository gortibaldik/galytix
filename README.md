# Assignment

## How to Run

At first, the data should be put into `./data`. For some reason, anytime I tried to use `gdown` for downloading vectors data from google, I got the exception that too many users try to download this data. The code tries to download the file, however if it fails then the application won't start.

The `./data` should contain files specified in [`config.py`](./embedding_engine/config.py).

The solution should be runnable with `docker compose`, please, use the following command to run it:

```
sudo docker compose up --build
```

## TODO

- [ ] Automatic download of GoogleNews-vectors
- [ ] Threaded execution of extractions to vectors.csv to see the progress
