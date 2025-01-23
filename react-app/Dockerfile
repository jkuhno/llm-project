FROM node

WORKDIR /usr/src/app

COPY package.json .
RUN npm i

COPY . /usr/src/app/

## EXPOSE [Port you mentioned in the vite.config file]

EXPOSE 5173

CMD ["npm", "run", "dev"]