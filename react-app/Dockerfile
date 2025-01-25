FROM node:18 AS base
WORKDIR /usr/src/app
COPY package.json ./
RUN npm install
COPY . .

FROM base AS dev
EXPOSE 5173
CMD ["npm", "run", "dev"]

FROM base AS prod
ARG VITE_APP_BACKEND_ADDRESS
ENV VITE_APP_BACKEND_ADDRESS $VITE_APP_BACKEND_ADDRESS
RUN echo "Backend Address: $VITE_APP_BACKEND_ADDRESS" && npm run build

# Step 2: Serve with Nginx
FROM nginx:latest
COPY --from=prod /usr/src/app/dist /usr/share/nginx/html
COPY nginx.conf  /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]