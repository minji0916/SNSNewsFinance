from database import get_decoded_summaries

# DB에 연결해서 null값이 아닌 summary와 news_id 가져옴 (json 디코딩 해서 가져옴)
# summary 값이 있어야 test 동작함
summaries = get_decoded_summaries()
print(f"summaries 구조확인\n : {summaries[:3]} \n")
print(f"news_id : {summaries[0]['news_id']} \n")
print(f"summaries[0] 구조 확인\n : {summaries[0]['summary']} \n")
print(f"point_1\n : {summaries[0]['summary']['point_1']} \n")
print(f"point_2\n : {summaries[0]['summary']['point_2']} \n")
print(f"point_3\n : {summaries[0]['summary']['point_3']} \n")
print(f"insight\n : {summaries[0]['summary']['insight']} \n")


###############################
""" 그냥 json 값 디코딩 하는 코드 : 아직 db에 summary 값 없을 때, 사용"""
import json

# News 테이블에서 저장된 summary 값 중 하나로, JSON 문자열입니다.
json_string = '{"point_1": "SK\uc774\ub178\ubca0\uc774\uc158\uacfc SK E&S\uc758 \ud569\ubcd1\uc548\uc774 \uc8fc\uc8fc\ucd1d\ud68c\uc5d0\uc11c \uc555\ub3c4\uc801\uc778 \ucc2c\uc131\uc73c\ub85c \ud1b5\uacfc\ub418\uc5c8\ub2e4.", "point_2": "\ud569\ubcd1\uc5d0 \ubc18\ub300\ud558\ub294 \uc8fc\uc8fc\ub4e4\uc774 \ud589\uc0ac\ud560 \uc8fc\uc2dd\ub9e4\uc218\uccad\uad6c\uad8c \uaddc\ubaa8\uac00 \ud5a5\ud6c4 \ud569\ubcd1 \uc0ac\uc5c5 \uc7ac\ud3b8 \uc791\uc5c5\uc5d0 \uc911\uc694\ud55c \ubcc0\uc218\ub85c \uc791\uc6a9\ud560 \uc804\ub9dd\uc774\ub2e4.", "point_3": "\uad6d\ubbfc\uc5f0\uae08 \ub4f1 \ud569\ubcd1\uc5d0 \ubc18\ub300\ud558\ub294 \uc8fc\uc8fc\ub4e4\uc774 \uc8fc\uc2dd\ub9e4\uc218\uccad\uad6c\uad8c\uc744 \uc804\ub7c9 \ud589\uc0ac\ud55c\ub2e4\uba74 SK\uc774\ub178\ubca0\uc774\uc158\uc740 8000\uc5b5\uc6d0 \uc900\ube44\uc561\uc744 \ucd08\uacfc\ud574\uc57c \ud558\ub294 \uc0c1\ud669\uc5d0 \uc9c1\uba74\ud560 \uc218 \uc788\ub2e4.", "insight": "SK\uc774\ub178\ubca0\uc774\uc158\uacfc SK E&S\uc758 \ud569\ubcd1 \uc131\uacf5 \uc5ec\ubd80\ub294 \uc8fc\uc2dd\ub9e4\uc218\uccad\uad6c\uad8c \ud589\uc0ac \uaddc\ubaa8\uc640 \uad6d\ubbfc\uc5f0\uae08 \ub4f1 \ubc18\ub300 \uc8fc\uc8fc\uc758 \uc785\uc7a5 \ubcc0\ud654\uc5d0 \ub2ec\ub824\uc788\ub2e4."}'

# JSON 문자열을 Python 객체로 변환
summaries = json.loads(json_string)

# 이제 decoded_data는 Python 딕셔너리입니다.
print("="*50)
print("그냥 json 값 디코딩 하는 코드")
print(f"point_1\n : {summaries['point_1']} \n")
print(f"point_2\n : {summaries['point_2']} \n")
print(f"point_3\n : {summaries['point_3']} \n")
print(f"insight\n : {summaries['insight']} \n")




