# ============================================
# 🔥 FLIR Lepton 3.5 기본 테스트 코드
# 만든이: 송둥윤 (불사대마왕)
# 기능: 라즈베리파이에서 Lepton 3.5 영상 출력 테스트
# ============================================

from pylepton import Lepton
import numpy as np
import cv2
import time

print("🔥 Lepton 3.5 테스트 시작 중...")

# Lepton 기본 포트(/dev/spidev0.0)로 열기
with Lepton() as lepton:
    while True:
        # 프레임 캡처
        img, _ = lepton.capture()

        # 값 범위 정리 (0~65535 → 8비트)
        img = np.clip(img, 0, 65535)
        img8 = (img / 256).astype(np.uint8)

        # 보기 쉽게 컬러맵 적용
        thermal = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)

        # 화면에 표시
        cv2.imshow("Lepton 3.5 Thermal View", thermal)

        # q 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("❄️ 프로그램 종료")
            break

cv2.destroyAllWindows()
