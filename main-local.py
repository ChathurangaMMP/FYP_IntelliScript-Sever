
import chainlit as cl
import requests


# @cl.app
# def app():
#     prompt = cl.text_input("Enter your prompt:")
#     if cl.button("Generate"):
#         response = requests.post(
#             "http://192.248.10.43:5000/generate_text", json={"prompt": prompt})
#         cl.text(response.json()["response"])


# if __name__ == "__main__":
#     app.run()


@cl.on_chat_start
async def start():
    msg = cl.Message(content="Starting the bot.....")
    await msg.send()
    msg.content = "Hi, Welcome to the IntelliScript Bot. What is your query?"
    await msg.update()


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    res = requests.post(
        "http://192.248.10.43:5000/generate_text", json={"prompt": message.content})
    answer = res
    # sources = res["source_documents"]

    # if sources:
    #     answer += f"\nSources:" + str(sources)
    # else:
    #     answer += f"\nNo Sources Found"

    await cl.Message(content=answer).send()
