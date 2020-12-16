void setup()
{
  Serial.begin(9600);
  pinMode(3, OUTPUT);
  pinMode(12, OUTPUT);
}

void loop()
{
  while (true) {
     if(Serial.available() > 0) {
      byte num = Serial.read() - 0x30;
      Serial.println(num);
      if (num == 1) {
        digitalWrite(3, HIGH);
      } else {
        digitalWrite(12, HIGH);
      }
      delay(5000);
      digitalWrite(3, LOW);
      digitalWrite(12, LOW);
    }
  }
}
  
