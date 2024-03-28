#include <Wire.h>
#include <Adafruit_BMP280.h>
#include <InfluxDbClient.h>
#include <InfluxDbCloud.h>
#include <WiFiMulti.h>
#include <DHT.h>

// WiFi AP SSID
#define WIFI_SSID "Wifi SSID"
// WiFi password
#define WIFI_PASSWORD "WiFi password"
// InfluxDB v2 server url
#define INFLUXDB_URL "Bucket URL"
// InfluxDB v2 server or cloud API authentication token
#define INFLUXDB_TOKEN "Bucket-Token"
// InfluxDB v2 organization id
#define INFLUXDB_ORG "Bucket ORG"
// InfluxDB v2 bucket name
#define INFLUXDB_BUCKET "Bucket Name"

#define TZ_INFO "IST-5:30" 

Adafruit_BMP280 bmp; // Declare the BMP280 object
DHT dht_sensor(23, DHT22); // Declare the DHT22 object with pin 23

WiFiMulti wifiMulti;
InfluxDBClient client(INFLUXDB_URL, INFLUXDB_ORG, INFLUXDB_BUCKET, INFLUXDB_TOKEN, InfluxDbCloud2CACert);

void setup() {
  Serial.begin(115200);

  // Setup wifi
  WiFi.mode(WIFI_STA);
  wifiMulti.addAP(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to wifi");
  while (wifiMulti.run() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println();
  
  // Initialize BMP280 sensor
  if (!bmp.begin(0x76)) {
    Serial.println("Could not find a valid BMP280 sensor, check wiring!");
    while (1);
  }

  // Initialize DHT sensor
  dht_sensor.begin();

  // Accurate time is necessary for certificate validation and writing in batches
  timeSync(TZ_INFO, "pool.ntp.org", "time.nis.gov");

  // Check server connection
  if (client.validateConnection()) {
    Serial.print("Connected to InfluxDB: ");
    Serial.println(client.getServerUrl());
  } else {
    Serial.print("InfluxDB connection failed: ");
    Serial.println(client.getLastErrorMessage());
  }
}

void loop() {
  // Get latest sensor readings
  float bmp_pressure = bmp.readPressure() / 100.0F; // Convert pressure to hPa
  float bmp_altitude = bmp.readAltitude(1013.25);

  // Read DHT22 sensor
  float dht_temperature = dht_sensor.readTemperature();
  float dht_humidity = dht_sensor.readHumidity();

  // Create data points for BMP280 and DHT22
  Point sensorReadings("bmp280 and dht22");
  sensorReadings.addTag("location", "Lab");
  sensorReadings.addField("altitude", bmp_altitude);
  sensorReadings.addField("pressure", bmp_pressure);
  sensorReadings.addField("temperature", dht_temperature);
  sensorReadings.addField("humidity", dht_humidity);

  
  // Write the points into the buffer
  client.writePoint(sensorReadings);

  // If there is no Wifi signal, try to reconnect
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wifi connection lost");
  }

  // Wait for 10 seconds
  Serial.println("Waiting for 10 seconds");
  delay(10000);
}
