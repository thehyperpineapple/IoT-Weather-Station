import { InfluxDBClient, Point } from '@influxdata/influxdb3-client';
import { config as configDotenv } from 'dotenv';
import { Router, Request, Response } from 'express';

configDotenv();

export const fetchData = Router();

const token = process.env.INFLUX_TOKEN;
let database = process.env.DATABASE;
const client = new InfluxDBClient({ host: 'https://us-east-1-1.aws.cloud2.influxdata.com', token: token });

async function queryData() {
    const query = `SELECT * FROM "measurements" WHERE time >= now() - interval '30 days' AND ("temperature" IS NOT NULL OR "pressure" IS NOT NULL) LIMIT 100`;
    const rows = await client.query(query, database);

    // Process the rows as a stream
    const values: any[] = [];
    let count = 0;
    for await (const value of rows) {
        values.push(value);
        count++;
        if (count >=100) {
          break
        }
    }
    const json = JSON.stringify(values);
    return json;
}

fetchData.get('/fetchData', async (req: Request, res: Response) => {
    try {
        const json = await queryData();
        res.send(json);
        client.close();
        console.log(`[server]: Server on route /fetchData`);
    } catch (error) {
        console.error('Error fetching data:', error);
        res.status(500).send('Internal Server Error');
    }
});


