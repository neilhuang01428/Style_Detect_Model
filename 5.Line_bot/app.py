# encoding: utf-8
# 20190815 created by ms lyu

from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage,VideoSendMessage,ImageSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    TemplateSendMessage, ConfirmTemplate, MessageAction,
    ButtonsTemplate, ImageCarouselTemplate, ImageCarouselColumn, URIAction,
    PostbackAction, DatetimePickerAction,
    CameraAction, CameraRollAction, LocationAction,
    CarouselTemplate, CarouselColumn, PostbackEvent,
    StickerMessage, StickerSendMessage, LocationMessage, LocationSendMessage,
    ImageMessage, VideoMessage, AudioMessage, FileMessage,
    UnfollowEvent, FollowEvent, JoinEvent, LeaveEvent, BeaconEvent,
    FlexSendMessage, BubbleContainer, ImageComponent, BoxComponent,
    TextComponent, SpacerComponent, IconComponent, ButtonComponent,
    SeparatorComponent, QuickReply, QuickReplyButton
)
import tempfile, os

# load config info
from config.config import CHANNEL_SECRET, CHANNEL_ACCESS_TOKEN, DOMAIN_URL
from style.style import predict
from template.dashboard import Dashboard
from datetime import datetime,timezone,timedelta
import threading
sem = threading.Semaphore()

# img tmp folder
static_tmp_path = os.path.join(os.getcwd(), 'static', 'tmp')
static_processed_path = os.path.join(os.getcwd(), 'static', 'processed')

# static url
app = Flask(__name__, static_url_path = "/images" , static_folder = "./static/")

handler = WebhookHandler(CHANNEL_SECRET) 
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN) 

# User Info 
USER_ID=''
USER_NAME=''

@app.route('/')
def index():
    return "<p>Emotion Recognition Bot is running!</p>"

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

# =================================
#  Handle Text event
# =================================

@handler.add(MessageEvent, message=TextMessage)  # default
def handle_text_message(event):                 
    msg = event.message.text #message from user

    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text='哈囉~請傳一張圖片給我'))
        

# =================================
#  Handle Image event
# =================================

@handler.add(MessageEvent, message=ImageMessage)  
def handle_img_message(event):
    if isinstance(event.source, SourceUser) or isinstance(event.source, SourceGroup) or isinstance(event.source, SourceRoom):
        profile = line_bot_api.get_profile(event.source.user_id)
        USER_ID = profile.user_id
        USER_NAME = profile.display_name

    if isinstance(event.message, ImageMessage):
        ext = 'jpg'
        # date
        dt = datetime.utcnow()
        dt = dt.replace(tzinfo=timezone.utc)
        tzutc_8 = timezone(timedelta(hours=8))
        local_dt = dt.astimezone(tzutc_8)
        mdatetime = local_dt.strftime('%Y%m%d')
    
        message_content = line_bot_api.get_message_content(event.message.id)
        with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix=mdatetime+'-',delete=False) as tf:
            for chunk in message_content.iter_content():
                tf.write(chunk)
            tempfile_path = tf.name

        dist_path = tempfile_path + '.' + ext

        #return file name
        dist_name = os.path.basename(dist_path)
        
        os.rename(tempfile_path, dist_path)

        # Start predict
#        img = Emotion(dist_path)
        resp = predict(dist_path)
        print(resp)

        # 判斷回傳臉數
        numOfPeople = resp["numOfPeople"]
        numOfObject = resp["numOfObject"]

        dashboard = Dashboard()

        if numOfObject ==0 and numOfPeople ==0:
            img_url = DOMAIN_URL+'/images/tmp/{0}'.format(dist_name)
        else:
            img_url = DOMAIN_URL+'/images/processed/{0}'.format(resp["pic_name"])
        sem.acquire()
        if numOfPeople>0 and type(numOfObject)==list:  #有人也有object
            line_bot_api.reply_message(event.reply_token, dashboard.people_object(img_url,resp))
        else:
            #沒人也沒object
            if numOfObject ==0 and numOfPeople ==0:
                line_bot_api.reply_message(event.reply_token, dashboard.no_peole_no_object(img_url,resp))
            #有人但沒object
            elif numOfObject==0 and numOfPeople!=0:
                line_bot_api.reply_message(event.reply_token, dashboard.people_no_object(img_url,resp))
            #沒人但有object
            else:
                line_bot_api.reply_message(event.reply_token, dashboard.no_people_object(img_url,resp))
        sem.release()

#        if numOfFaces == 0:
#            line_bot_api.reply_message(event.reply_token, dashboard.no_face())
#        # multiple face
#        elif numOfFaces > 1:
#            line_bot_api.reply_message(event.reply_token, dashboard.too_many_faces())
#        else:
#
#            img_url = DOMAIN_URL+'/images/alignment/{0}'.format(dist_name)
#
#            line_bot_api.reply_message(event.reply_token, dashboard.show_single_result(img_url, resp["emotion"], resp["confidence"]))
#
#        line_bot_api.reply_message(event.reply_token,
#TextSendMessage(text="{0}\n{1}\n{2}\n{3}\n{4}".format(resp["info"], resp["style"], resp["style_confidence"],resp["object"],resp["object_prob"])))

        # list data in tmp folder
        # with os.scandir(static_tmp_path) as entries:
        #     for entry in entries:
        #         print(entry.name)

        # remove
#        os.remove(dist_path)


# =================================
#  Handle Welcome event
# =================================

@handler.add(FollowEvent)
def handle_join(event):
    if isinstance(event.source, SourceUser) or isinstance(event.source, SourceGroup) or isinstance(event.source, SourceRoom):
        profile = line_bot_api.get_profile(event.source.user_id)
        USER_ID = profile.user_id
        USER_NAME = profile.display_name

        # welcome message
        dashboard = Dashboard()

        line_bot_api.reply_message(
        event.reply_token,[
            dashboard.intro(USER_NAME)
            ]
        )

# =================================
#  Basic Function
# =================================

#push text
#def line_single_push(id, txt):
#    line_bot_api.push_message(id,
#        TextSendMessage(text=txt))
#
##push sticker
#def line_single_sticker(id, packed_id, sticker_id):
#    line_bot_api.push_message(id,
#        StickerSendMessage(package_id=packed_id,
#    sticker_id=sticker_id))
#
##push img
#def line_single_img(id, preview, orign):
#    line_bot_api.push_message(id,
#        ImageSendMessage(
#    original_content_url=orign,
#    preview_image_url=preview
#))

import os
if __name__ == "__main__":
    # heroku
    #app.run(host='0.0.0.0', port=os.environ['PORT'])
    # local test
    app.run(host='0.0.0.0', port=2000)
