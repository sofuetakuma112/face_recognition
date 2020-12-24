void setup()
{
  Serial.begin(9600);
  pinMode(3, OUTPUT);
  pinMode(8, OUTPUT);
  pinMode(12, OUTPUT);
}

void loop()
{
  while (true) {
     if(Serial.available() > 0) {
      byte num = Serial.read() - 0x30;
      if (num == 1) {  // OK
        digitalWrite(3, HIGH);
        delay(5000);
      } else if (num == 2) {
        digitalWrite(12, HIGH);
      } else if (num == 3) {
        digitalWrite(12, LOW);
      } else {  // NO
        digitalWrite(8, HIGH);
        delay(5000);
      }
      digitalWrite(3, LOW);
      digitalWrite(8, LOW);
    }
  }
}
