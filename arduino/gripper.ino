#include <Servo.h>

Servo gripServo;

const int SERVO_PIN = 9;
int currentAngle = 0;

void setup() {
  gripServo.attach(SERVO_PIN);
  gripServo.write(0);
  Serial.begin(9600);
  Serial.println("READY");
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  String command = Serial.readStringUntil('\n');
  command.trim();

  if (command == "OPEN") {
    smoothMove(0);
  } else if (command.startsWith("CLOSE:")) {
    int target = command.substring(6).toInt();
    target = constrain(target, 0, 90);
    smoothMove(target);
  } else if (command == "STATUS") {
    Serial.println("ANGLE:" + String(currentAngle));
  } else if (command == "PING") {
    Serial.println("PONG");
  }
}

void smoothMove(int target) {
  Serial.println("MOVING");

  if (target == currentAngle) {
    Serial.println("DONE");
    return;
  }

  int step = target > currentAngle ? 1 : -1;
  while (currentAngle != target) {
    currentAngle += step;
    gripServo.write(currentAngle);
    delay(12);
  }

  Serial.println("DONE");
}
