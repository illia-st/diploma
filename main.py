from typing import Final
from telegram import Message, Update, constants
import telegram
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import RobertaTokenizer, RobertaForSequenceClassification

TOKEN: Final = '6919031911:AAFEzbmL0YU2hjYM7u6mQMDYK-jcd3x-T1I'
BOT_USERNAME: Final = '@news_antifake_bot'

from dotenv import load_dotenv
import os

load_dotenv()

TOKEN: Final = os.getenv('TOKEN')
BOT_USERNAME: Final = os.getenv('BOT_USERNAME')

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply_text = """
Привіт! Я детектор ІПСО.
Перешліть мені новину з іншого новинного каналу і я дам вам свою думку.
Результати ваших запитів можуть містити помилки і не потрібно завжди покладатись на них.
Робіть додаткову перевірку новин, які ви надсилаєте, а цей інстурмент може стати вам в нагоді, як
детектор ПОТЕНЦІЙНОЇ психологічної атаки.
P.S: Я розумію лише українську мову, тому на новини російською можу давати досить не точну відповідь.
    """
    await update.message.reply_text(reply_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply_text = """
Перешліть мені новину з іншого новинного каналу і я дам вам свою думку.
Результати ваших запитів можуть містити помилки і не потрібно завжди покладатись на них.
Робіть додаткову перевірку новин, які ви надсилаєте, а цей інстурмент може стати вам в нагоді, як
детектор ПОТЕНЦІЙНОЇ психологічної атаки.
P.S: Я розумію лише українську мову, тому на новини російською можу давати досить не точну відповідь.
    """
    await update.message.reply_text(reply_text)

def define_response(probs):
    not_psyops = probs.data[0][0]
    psyops = probs.data[0][1]
    pred_label_idx = probs.argmax()
    if pred_label_idx == 0:
        return f'З ймовірністю {not_psyops:.3f} це не ІПСО'
    else:
        return f'З ймовірністю {psyops:.3f} це ІПСО'

#  Handle responses
def predict(text: str) -> int:
    tokenizer_fine_tuned = RobertaTokenizer.from_pretrained('../diploma')
    model_fine_tuned = RobertaForSequenceClassification.from_pretrained('../diploma')
    predict_input = tokenizer_fine_tuned.encode(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )

    output = model_fine_tuned(predict_input)

    probs = output[0].softmax(1)

    return define_response(probs)

def get_text_from_message(message: Message):
    if message.caption is None:
        return message.text
    return message.caption

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat_type = message.chat.type
    error_messsage = '''
Даний тип повідомлень не підтримується.
Для аналізу новин надішілсть, будь ласка, новинний пост з будь-якої групи.
'''
    if chat_type == 'private' and message.forward_origin and message.forward_origin.type == constants.MessageOriginType.CHANNEL:
        if isinstance(message.forward_origin, telegram.MessageOriginChannel):
            text_to_predict = get_text_from_message(message)
            prediction = predict(text_to_predict)
            await update.message.reply_text(prediction)
        else:
            print("The forward_origin is not a MessageOriginChannel")
    else:
        await update.message.reply_text(error_messsage)

async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

if __name__ == '__main__':
    print('Starting bot')
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))

    app.add_handler(MessageHandler(filters.ALL, handle_message))

    app.add_error_handler(error)

    print('Polling...')
    app.run_polling(poll_interval=3)
