import requests
import json


def generate(prompt):
    print(f"prompt: {prompt}")
    response = requests.post('http://ollama.n705.work/api/chat',
                             # "http://ollama.service.svc.cluster.local:11434/api/chat"
                             # curl -fsSL https://ollama.com/install.sh | sh
                             json={"model": "llama3.3",  # "deepseek-r1:14b", "llama3.2"
                                   "messages": prompt,
                                   "stream": True
                                   },
                             auth=('n705', 'ADugPA3o18d9ocfE'))
    response.raise_for_status()
    output = ""
    for line in response.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
            # the response streams one token at a time, print that as we receive it
            # print(content, end="", flush=True)

        if body.get("done", False):
            response = output
            break
    print(f"\nresponse: {response}")


def main():
    test_prompt = [{
        "role": "user",
        "content": "Valid actions: sleep, eat, attack, chop, drink, place, make, mine. You are a player playing a game. Suggest the best actions the player can take based on the things you see and the items in your inventory. Only use valid actions and objects.\nYou see water, grass, cow, and diamond. You are targeting grass. You have in your inventory plant. What do you do?"
    }]
    print()
    # generate(messages)
    generate(test_prompt)
    print()


if __name__ == "__main__":
    main()