FROM node:20-alpine

WORKDIR /app

# 装依赖（独立层，利用缓存）
COPY package.json package-lock.json* ./
RUN npm install

# 拷贝源码
COPY . .

EXPOSE 5174

# vite dev server，--host 暴露给宿主机访问
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5174"]
