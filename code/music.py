import pygame

# 初期化
pygame.init()
pygame.mixer.init()

# MIDIファイル読み込み
pygame.mixer.music.load("music/praeludium1_1.mid")

# 再生
pygame.mixer.music.play()

# 終わるまで待機
while pygame.mixer.music.get_busy():
    continue

pygame.quit()
