#include <Wire.h>
#include <SPI.h>
#include <Adafruit_BMP280.h>
#include <DHT.h>

#define BMP_SCK  (13)
#define BMP_MISO (12)
#define BMP_MOSI (11)
#define BMP_CS   (10)

#define DHT_SENSOR_PIN  23
#define DHT_SENSOR_TYPE DHT22

Adafruit_BMP280 bmp; // BMP280 sensor
DHT dht_sensor(DHT_SENSOR_PIN, DHT_SENSOR_TYPE); // DHT22 sensor

void setup() {
  Serial.begin(9600);
  while (!Serial)
    delay(100); // wait for native USB

  Serial.println(F("BMP280 and DHT22 Sensor Integration"));

  // BMP280 setup
  unsigned status = bmp.begin(0x76);
  if (!status) {
    Serial.println(F("Could not find a valid BMP280 sensor, check wiring or try a different address!"));
    while (1)
      delay(10);
  }
  bmp.setSampling(Adafruit_BMP280::MODE_NORMAL, Adafruit_BMP280::SAMPLING_X2,
                  Adafruit_BMP280::SAMPLING_X16, Adafruit_BMP280::FILTER_X16,
                  Adafruit_BMP280::STANDBY_MS_500);

  // DHT22 setup
  dht_sensor.begin();
}

void readBMP280() {
  Serial.print(F("BMP280 - Pressure = "));
  Serial.print(bmp.readPressure() / 100.0F);
  Serial.println(" Pa");

  Serial.print(F("BMP280 - Approx altitude = "));
  Serial.print(bmp.readAltitude(1013.25)); /* Adjusted to local forecast! */
  Serial.println(" m");

}

void readDHT22() {
  float humi = dht_sensor.readHumidity();
  float tempC = dht_sensor.readTemperature();
  float tempF = dht_sensor.readTemperature(true);

  if (isnan(tempC) || isnan(tempF) || isnan(humi)) {
    Serial.println("Failed to read from DHT sensor!");
  } else {
    Serial.print("DHT22 - Humidity: ");
    Serial.print(humi);
    Serial.println("%");

    Serial.print("DHT22 - Temperature: ");
    Serial.print(tempC);
    Serial.print("°C  ~  ");
    Serial.print(tempF);
    Serial.println("°F");
  }

  Serial.println();
}

void loop() {
  readBMP280();
  readDHT22();
  delay(2000);
}
