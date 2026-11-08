"""입력 소스: 프레임 이미지 + FrameInfo를 만든다. 결과는 모두 같은 FrameInfo 형식이라 뒤(검출기·pipeline)는 소스를 모른다.

    images.py     사진 폴더 (K·자세를 모르면 가정값, source="assumed")
    recording.py  아이폰 녹화본 폴더 (frames.jsonl의 K·자세, source="device"). 영상 풀기에 OpenCV 필요
"""
