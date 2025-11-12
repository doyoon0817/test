# ============================================
#  🔥 스마트 냉난방 시스템 (불사대마왕팀)
#  Comfort Never Dies.
#  구성: FLIR Lepton 3.5 + Breakout + PIR Sensor + Raspberry Pi
# ============================================

from gpiozero import MotionSensor
from pylepton import Lepton
import numpy as np
import cv2
import time

# === PIR 센서 설정 ===
# PIR 센서 OUT → GPIO17, VCC → 5V or 3.3V, GND → GND
pir = MotionSensor(17)

print("🔥 불사대마왕 시스템 부팅 중...")
print("Lepton 3.5 열화상 스트리밍 시작!")
print("PIR 감지 시 온도 및 위치 표시")

# === Lepton 열화상 캡처 루프 ===
with Lepton() as lepton:
    while True:
        # 열화상 프레임 수신
        img, _ = lepton.capture()
        img = np.clip(img, 0, 65535)
        img8 = (img / 256).astype(np.uint8)

        # 컬러맵 적용 (보기 쉽게)
        thermal = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)

        # PIR 감지되면 온도/위치 분석
        if pir.motion_detected:
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img)
            temp_c = maxVal / 10.0  # 섭씨 온도 추정

            print(f"👤 움직임 감지됨 | 최고온도: {temp_c:.1f} °C | 위치: {maxLoc}")

            # 화면 표시
            cv2.circle(thermal, maxLoc, 5, (255, 255, 255), 2)
            cv2.putText(
                thermal,
                f"{temp_c:.1f}C",
                (maxLoc[0] + 10, maxLoc[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # 열화상 영상 출력
        cv2.imshow("Lepton 3.5 Thermal View", thermal)

        # 'q' 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("시스템 종료 중...")
            break

cv2.destroyAllWindows()
