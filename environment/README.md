# Brev에서 실습하기

1. 현장에서 안내받은 링크나 QR 코드로 Brev에 로그인합니다. 휴대폰으로 초대를 수락했다면 실습할 컴퓨터에서도 같은 계정을 사용하세요.
2. 행사 페이지에서 **Deploy Launchable**을 한 번 누릅니다. [Brev 콘솔](https://brev.nvidia.com/)에서 내 인스턴스 상태가 **Running**으로 바뀌면 **JupyterLab**을 엽니다.
3. [00_Start_Here.ipynb](../00_Start_Here.ipynb)에서 첫 셀을 실행합니다. 파일 목록이 먼저 보이면 `PyTorch-Day-Korea-2026-PhysicsNeMo` 폴더에서 이 노트북을 여세요. 커널은 **AI4Sci PhysicsNeMo 2.2.2 (uv / CUDA)**입니다.
4. 실습이 끝나면 노트북을 저장합니다. Brev 콘솔에서 내 인스턴스를 선택하고 **Stop**을 누른 뒤 **Stopped** 상태를 확인하세요.

## 별도 GPU VM에서 설치

NVIDIA 드라이버와 Brev의 호스트 JupyterLab이 준비된 Linux GPU VM에서 JupyterLab 터미널을 열고 아래 명령을 실행합니다. 설치는 JupyterLab을 실행한 사용자 계정으로 진행하세요.

```bash
git clone https://github.com/yang926/PyTorch-Day-Korea-2026-PhysicsNeMo.git
cd PyTorch-Day-Korea-2026-PhysicsNeMo
python3 brev/install.py
```

설치 프로그램은 Python 3.12.11, PyTorch 2.10.0+cu128, PhysicsNeMo 2.2.2를 사용하는 커널을 등록하고 FNO 실습 데이터를 준비합니다. 전체 패키지 버전은 [CUDA 환경 패키지 목록](linux-cu128.lock.txt)에서 확인할 수 있습니다.
