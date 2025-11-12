from pylepton import Lepton
import numpy as np

print("🔥 Lepton 테스트 시작...")

with Lepton() as lepton:
    img, _ = lepton.capture()
    print("프레임 크기:", img.shape)
    print("최대값:", np.max(img))
    print("최소값:", np.min(img))

print("✅ Lepton 데이터 수신 완료!")
