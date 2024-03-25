import * as express from "express";
import { defaultRoute } from "./defaultRoute";
import { fetchData } from "./fetchData";


export const routes = express.Router();

routes.use(defaultRoute);
routes.use(fetchData);