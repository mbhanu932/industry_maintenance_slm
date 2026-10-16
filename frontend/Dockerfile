# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Runtime stage
FROM node:18-alpine

WORKDIR /app

# Install serve to run the built app
RUN npm install -g serve

# Copy built application from builder
COPY --from=builder /app/dist /app/dist

# Expose port
EXPOSE 5173

# Serve the application
CMD ["serve", "-s", "dist", "-l", "5173"]
