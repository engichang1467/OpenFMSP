# OpenFMSP
The reimplementation of foundation model self-play 

[Check out the original paper here](https://arxiv.org/pdf/2507.06466)

## Set up

1.  Install the important packages/libraries

```
pip install -r requirements.txt
```

2. Set up your OpenAI API credentials in a `.env` file

```
export OPENAI_API_KEY="sk-proj-..."
```

3. Export the credentials

```
source .env
```

4. Run the training script

```
python main.py
```