# BFMC - Embedded platform project

The project contains all the software present on the Nucleo board, together with the documentation on how to create new components and what are the features of the given one. Some of the feature are:
- Communication protocol between RPi and Nucleo,
- Motors control,
- IMU readings
- Notifications from Power Board
- Architecture prone to features addition

## The documentation is available in details here:
[Documentation](https://bosch-future-mobility-challenge-documentation.readthedocs-hosted.com/data/embeddedplatform.html) 

Lưu ý: Repo embed này đã được custom để thêm vào 1 state mới trong robotstatemachine (state 5) để nhập lệnh vào pwm và điều khiển trực tiếp bằng dữ liệu pwm thay vì speed

- Add thêm phần điều khiển pwm để steer cho servo.
