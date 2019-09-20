
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,VideoSendMessage,
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


class Dashboard():
    def __init__(self):
        pass
    
    def no_peole_no_object(self,imgurl,resp):
        bubble = BubbleContainer(
                                 # header
                                 header=BoxComponent(
                                                     layout='horizontal',
                                                     contents=[
                                                               TextComponent(text='辨識結果', size='lg', color='#5b5b5b')
                                                               ]
                                                     ),
                                 
                                 #hero
                                 hero=ImageComponent(url='{0}'.format(imgurl), size='3xl', aspect_ratio='1:1', aspect_mode='fit'),
                                 
                                 # body
                                 body=BoxComponent(
                                                   layout='vertical',
                                                   spacing='md',
                                                   contents=[
                                                             
                                                             SeparatorComponent(margin='md'),
                                                             
                                                             BoxComponent(
                                                                          layout = 'horizontal',
                                                                          spacing='md',
                                                                          contents = [
                                                                                      
                                                                                      TextComponent(text='很抱歉, 圖片未偵測到人物,\n 請考慮環境因素影響(如燈光)、角度、距離等因素，麻煩您再次嘗試。\n\nPlease try again. Thanks for your cooperation.', size='sm', wrap=True, color='#9d9d9d', align='center')
                                                                                      ]
                                                                          )
                                                             ]
                                                   )
                                 )

        message = FlexSendMessage(alt_text="辨識結果", contents=bubble)
        return message
    
    
    def people_no_object(self,imgurl,resp):
        bubble = BubbleContainer(
                                 # header
                                 header=BoxComponent(
                                                     layout='horizontal',
                                                     contents=[
                                                               TextComponent(text='辨識結果', size='lg', color='#5b5b5b')
                                                               ]
                                                     ),
                                 
                                 #hero
                                 hero=ImageComponent(url='{0}'.format(imgurl), size='3xl', aspect_ratio='1:1', aspect_mode='fit'),
                                 
                                 # body
                                 body=BoxComponent(
                                                   layout='vertical',
                                                   spacing='md',
                                                   contents=[
                                                             
                                                             SeparatorComponent(margin='md'),
                                                             
                                                             BoxComponent(
                                                                          layout = 'horizontal',
                                                                          spacing='md',
                                                                          contents = [
                                                                                      
                                                                                      TextComponent(text='很抱歉, 雖然有偵測到對象,\n 但無法辨別服裝風格，請考慮環境因素影響(如燈光)、角度、距離等因素後，再次嘗試一次。\n\nPlease try again. Thanks for your cooperation.', size='sm', wrap=True, color='#9d9d9d', align='center')
                                                                                      ]
                                                                          )
                                                             ]
                                                   )
                                 )
            
        message = FlexSendMessage(alt_text="辨識結果", contents=bubble)
        return message
    def no_people_object(self,imgurl,resp):
        if resp["style"]==0:
            style = "正式服裝"
        elif resp["style"]==1:
            style ="商務休閒"
        else:
            style ="休閒"
        s_confidence = resp["style_confidence"]

        body_content_temp =[SeparatorComponent(margin='md'),
                            BoxComponent(
                                         layout = 'horizontal',
                                         spacing='md',
                                         contents = [
                                                     TextComponent(text='風格 :', size='lg', color='#9d9d9d', align='center'),
                                                     TextComponent(text='{0}'.format(style), size='lg', color='#00BE00', weight='bold')
                                                     ]
                                         ),
                            BoxComponent(
                                         layout = 'horizontal',
                                         spacing='md',
                                         contents = [
                                                     TextComponent(text='信心值 :', size='lg', color='#9d9d9d', align='center'),
                                                     TextComponent(text='{:.1f}%'.format(s_confidence*100), size='lg', color='#00BE00', weight='bold')
                                                     ]
                                         ),
                            BoxComponent(
                                         layout = 'horizontal',
                                         spacing='md',
                                         contents = [
                                                     TextComponent(text='色系 :', size='lg', color='#9d9d9d', align='center'),
                                                     TextComponent(text='{0}'.format(resp["color"]), size='lg', color='#00BE00', weight='bold')
                                                     ]
                                         ),
                            BoxComponent(
                                         layout = 'horizontal',
                                         spacing='md',
                                         contents = [
                                                     TextComponent(text='以下為偵測到的物件:', size='lg', color='#9d9d9d', align='center'),
                                                     ]
                                         ),
                            BoxComponent(layout = 'horizontal',spacing='md',contents = [TextComponent(text='物件', size='lg', color='#9d9d9d', align='center'),TextComponent(text='信心值', size='lg', color='#9d9d9d', weight='bold')])
                            ]
        print(resp["numOfObject"])
        print(len(resp["object"]))
        for i in range(len(resp["object"])):
            object_content_temp =[]
            object_content_temp.append(TextComponent(text='{0}'.format(resp["object"][i]), size='lg', color='#9d9d9d', align='center'))
            object_content_temp.append(TextComponent(text='{:.1f}%'.format(resp["object_prob"][i]*100), size='lg', color='#00BE00', weight='bold'))
            body_content_temp.append(BoxComponent(layout = 'horizontal',spacing='md',contents = object_content_temp))

        bubble = BubbleContainer(
                                 # header
                                 header=BoxComponent(
                                                     layout='horizontal',
                                                     contents=[
                                                               TextComponent(text='辨識結果', size='lg', color='#5b5b5b')
                                                               ]
                                                     ),
                                 
                                 #hero
                                 hero=ImageComponent(url='{0}'.format(imgurl), size='3xl', aspect_ratio='1:1', aspect_mode='fit'),
                                 
                                 # body
                                 body=BoxComponent(
                                                   layout='vertical',
                                                   spacing='md',
                                                   contents=body_content_temp
                                                   )
                                 )
        message = FlexSendMessage(alt_text="辨識結果", contents=bubble)
        return message
    
    def people_object(self,imgurl,resp):
        body_content_temp=[]
        for p in range(resp["numOfPeople"]):
            if resp["style"][p] == 0:
                style = "正式服裝"
            elif resp["style"][p] == 1:
                style = "商務休閒"
            else:
                style = "休閒"
            s_confidence = resp["style_confidence"][p]
            body_content_temp.append(SeparatorComponent(margin='md'))
            body_content_temp.append(BoxComponent(layout = 'horizontal',spacing='md',contents = [TextComponent(text='第%s位結果'%(p+1), size='lg', color='#444444', align='center', weight='bold')]))
            if resp["numOfObject"][p]>0:
                body_content_temp.append(BoxComponent(layout = 'horizontal',spacing='md',contents = [TextComponent(text='風格 :', size='lg', color='#9d9d9d',align='center'),TextComponent(text='{0}'.format(style), size='lg', color='#00BE00', weight='bold')]))
                body_content_temp.append(BoxComponent(layout = 'horizontal',spacing='md',contents = [TextComponent(text='信心值 :', size='lg', color='#9d9d9d', align='center'),TextComponent(text='{:.1f}%'.format(s_confidence*100), size='lg', color='#00BE00', weight='bold')]))
                body_content_temp.append(BoxComponent(layout = 'horizontal',spacing='md',contents =[TextComponent(text='色系 :', size='lg', color='#444444',align='center', weight='bold')]))
                body_content_temp.append(BoxComponent(layout = 'horizontal',spacing='md',contents = [TextComponent(text=resp["color"][p][0][0], size='lg', color='#9d9d9d',align='center'),TextComponent(text='{:.1f}%'.format(resp["color"][p][0][1]*100), size='lg', color='#00BE00', weight='bold')]))
                body_content_temp.append(BoxComponent(layout = 'horizontal',spacing='md',contents = [TextComponent(text=resp["color"][p][1][0], size='lg', color='#9d9d9d',align='center'),TextComponent(text='{:.1f}%'.format(resp["color"][p][1][1]*100), size='lg', color='#00BE00', weight='bold')]))
                body_content_temp.append(BoxComponent(layout = 'horizontal',spacing='md',contents = [TextComponent(text=resp["color"][p][2][0], size='lg', color='#9d9d9d',align='center'),TextComponent(text='{:.1f}%'.format(resp["color"][p][2][1]*100), size='lg', color='#00BE00', weight='bold')]))
                body_content_temp.append(BoxComponent(layout = 'horizontal',spacing='md',contents =[TextComponent(text='以下為偵測到的物件:', size='lg', color='#444444',align='center', weight='bold')]))
                body_content_temp.append(BoxComponent(layout = 'horizontal',spacing='md',contents = [TextComponent(text='物件', size='lg', color='#9d9d9d', align='center'),TextComponent(text='信心值', size='lg', color='#9d9d9d')]))
               

                for i in range(resp["numOfObject"][p]):
                    object_content_temp =[]
                    object_content_temp.append(TextComponent(text='{0}'.format(resp["object"][p][i]), size='lg', color='#9d9d9d', align='center'))
                    object_content_temp.append(TextComponent(text='{:.1f}%'.format(resp["object_prob"][p][i]*100), size='lg', color='#00BE00', weight='bold'))
                    body_content_temp.append(BoxComponent(layout = 'horizontal',spacing='md',contents = object_content_temp))
                                                                                                     
        bubble = BubbleContainer(
                                 # header
                                 header=BoxComponent(layout='horizontal',contents=[TextComponent(text='辨識結果', size='lg', color='#5b5b5b')]),
                                 #hero
                                 hero=ImageComponent(url='{0}'.format(imgurl), size='3xl', aspect_ratio='1:1', aspect_mode='fit'),
                                 # body
                                 body=BoxComponent(layout='vertical',spacing='md',contents=body_content_temp)
                                 )
        message = FlexSendMessage(alt_text="辨識結果", contents=bubble)
        return message
    

    # intro
    def intro(self, user_name):
        bubble = BubbleContainer(
            # header
            header=BoxComponent(
                layout='horizontal',
                contents=[
                    TextComponent(text='Hi {0},'.format(user_name), size='lg', color='#000000')
                ]
            ),

            # hero
            hero=ImageComponent(url='https://i.imgur.com/xYgkKHe.png', size='full', aspect_ratio='16:9', aspect_mode='cover'),

            # body
            body=BoxComponent(
                layout='vertical',
                spacing='md',
                contents=[
                    
                    SeparatorComponent(margin='md'),

                    BoxComponent(
                        layout = 'horizontal',
                        spacing='md',
                        contents = [
        
                            TextComponent(text='Hi 可以發一張圖片給我，\n我能識別出你的穿搭風格唷～\n\n', size='md', wrap=True, color='#272727', align='center')
                        ]
                    ),

                    BoxComponent(
                        layout = 'horizontal',
                        spacing='md',
                        contents = [
        
                            TextComponent(text='聲明\n辨識完即刪除，不會以任何形式保存圖片、人像等敏感性資訊', size='sm', wrap=True, color='#00BE00', align='center')
                        ]
                    )            

                ]
            )
        )
        message = FlexSendMessage(alt_text="hello~", contents=bubble)
        return message
