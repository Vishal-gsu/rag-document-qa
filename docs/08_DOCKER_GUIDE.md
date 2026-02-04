# 🐳 **08 - Docker Basics for Beginners**

**Don't know Docker? No problem!** This guide explains everything you need to know.

---

## **1. What is Docker? (Simple Explanation)**

### **The Problem Docker Solves**

Imagine you built an app that works perfectly on your computer, but when your friend tries to run it:
```
Friend: "It doesn't work!"
You: "But it works on my machine..."
Friend: "I get errors about missing libraries..."
```

**This is the classic "works on my machine" problem.**

### **Docker's Solution**

Docker packages your app + all its dependencies into a **container** that runs the same way everywhere.

```
Your Computer:        Friend's Computer:      Production Server:
┌──────────────┐     ┌──────────────┐        ┌──────────────┐
│  Container   │     │  Container   │        │  Container   │
│  ┌────────┐  │     │  ┌────────┐  │        │  ┌────────┐  │
│  │  App   │  │     │  │  App   │  │        │  │  App   │  │
│  │Dependencies│     │  │Dependencies│        │  │Dependencies│
│  └────────┘  │     │  └────────┘  │        │  └────────┘  │
└──────────────┘     └──────────────┘        └──────────────┘
     Works! ✅            Works! ✅               Works! ✅
```

**All three run identically** because the container includes everything needed.

---

## **2. Key Docker Concepts**

### **2.1 Docker Image**

Think of it like a **recipe** or **blueprint**.

```
Docker Image = Recipe for making a container
- Contains: Code, libraries, dependencies, settings
- Read-only (can't change it)
- Stored in Docker Hub (like GitHub for Docker)

Example:
"endeeio/endee-server:latest" = Official Endee recipe
```

**Real-world analogy:**
- Image = Cookie cutter mold 🍪
- Container = The actual cookie you bake

### **2.2 Docker Container**

A **running instance** of an image.

```
Docker Container = Actual running app
- Created from an image
- Has its own filesystem, network, processes
- Can be started, stopped, deleted
- Isolated from other containers

Example:
When you run "docker compose up", it creates a container from the Endee image
```

**Real-world analogy:**
- Image = House blueprint 📐
- Container = Actual house built from blueprint 🏠

You can build many houses (containers) from one blueprint (image).

### **2.3 Docker Volume**

**Persistent storage** that survives container restarts.

```
Problem: When you delete a container, all data inside is lost!

Solution: Docker Volume
- Stores data outside the container
- Data persists even if container is deleted
- Can be shared between containers

Example in our project:
endee-data volume = Where Endee stores your 4,772 vectors
```

**Real-world analogy:**
```
Container = Hotel room 🏨
Volume = Safe deposit box in the lobby 🔐

When you check out (delete container), your room is cleaned.
But your valuables in the safe (volume) remain!
```

### **2.4 Docker Compose**

A tool to manage **multiple containers** with one command.

```
docker-compose.yml = Configuration file
- Defines which containers to run
- Sets ports, volumes, environment variables
- Starts everything with one command

Instead of:
docker run -p 8080:8080 -v endee-data:/data ...  # Long command ❌

You do:
docker compose up -d  # One simple command ✅
```

---

## **3. Our Docker Setup (Endee Server)**

### **What We're Running**

```
Service: Endee Vector Database
Image: endeeio/endee-server:latest
Container Name: endee
Port: 8080 (your computer) → 8080 (container)
Volume: endee-data (persistent storage for vectors)
```

### **The docker-compose.yml File Explained**

```yaml
version: "3.8"                    # Docker Compose version

services:                         # List of containers to run
  endee:                          # Service name (you can call it anything)
    image: endeeio/endee-server:latest   # Which image to use
    container_name: endee         # Name of the running container
    ports:
      - "8080:8080"              # Port mapping (your PC:container)
    volumes:
      - endee-data:/data         # Volume mapping (persistent storage)
    environment:
      - ENDEE_PORT=8080          # Environment variable
      - ENDEE_HOST=0.0.0.0       # Allow connections from anywhere
    restart: always              # Auto-restart if it crashes

volumes:
  endee-data:                    # Define named volume
    driver: local                # Store on local disk
```

### **Port Mapping Explained**

```
"8080:8080" means:

8080 (left)  = Port on YOUR computer
  ↓
8080 (right) = Port INSIDE the container

When you visit: http://localhost:8080
→ Goes to port 8080 on your computer
→ Docker forwards it to port 8080 inside container
→ Endee server receives the request
```

**Example:**
```
You:           curl http://localhost:8080/status
                      ↓
Your PC:       Port 8080 (listening)
                      ↓ (Docker forwards)
Container:     Port 8080 (Endee server)
                      ↓
Response:      {"status": "ok"}
```

### **Volume Mapping Explained**

```
"endee-data:/data" means:

endee-data (left) = Named volume on your computer
  ↓
/data (right) = Directory INSIDE the container

When Endee saves vectors:
1. Writes to /data inside container
2. Docker stores it in endee-data volume
3. Data persists even if container is deleted!
```

**Where is the volume stored?**
```
Windows: C:\ProgramData\Docker\volumes\endee-data\_data
Linux:   /var/lib/docker/volumes/endee-data/_data
macOS:   /var/lib/docker/volumes/endee-data/_data
```

---

## **4. Essential Docker Commands**

### **Starting & Stopping**

```bash
# Start Endee (creates container if doesn't exist)
docker compose up -d
# -d = detached mode (runs in background)

# Stop Endee (keeps data!)
docker compose stop

# Restart Endee
docker compose restart

# Stop and remove container (data still safe in volume)
docker compose down

# ⚠️ DANGER: Stop and delete EVERYTHING including volume
docker compose down -v
# -v = remove volumes (deletes your 4,772 vectors!)
```

### **Checking Status**

```bash
# List running containers
docker ps

# Output:
CONTAINER ID   IMAGE                          STATUS        PORTS
abc123def456   endeeio/endee-server:latest   Up 5 minutes  0.0.0.0:8080->8080/tcp

# List all containers (including stopped)
docker ps -a

# Check if Endee is running
docker ps | grep endee
# If you see output → Running ✅
# If no output → Not running ❌
```

### **Viewing Logs**

```bash
# See what Endee is doing
docker logs endee

# Follow logs in real-time (like tail -f)
docker logs endee -f

# See last 50 lines
docker logs endee --tail 50

# Logs with timestamps
docker logs endee -t
```

**Example log output:**
```
[INFO] Endee server starting...
[INFO] HNSW index initialized
[INFO] Listening on 0.0.0.0:8080
[INFO] Server ready!
```

### **Inspecting Containers**

```bash
# Get detailed info about Endee container
docker inspect endee

# Get just the IP address
docker inspect endee | grep IPAddress

# Check volume mounts
docker inspect endee | grep -A 10 Mounts
```

### **Managing Volumes**

```bash
# List all volumes
docker volume ls

# Inspect endee-data volume
docker volume inspect endee-data

# Output shows:
{
    "Name": "endee-data",
    "Mountpoint": "/var/lib/docker/volumes/endee-data/_data",
    "Driver": "local"
}

# Backup volume (Linux/macOS)
docker run --rm -v endee-data:/data -v $(pwd):/backup ubuntu tar czf /backup/endee-backup.tar.gz /data

# Remove unused volumes (careful!)
docker volume prune
```

### **Executing Commands Inside Container**

```bash
# Open a shell inside the running container
docker exec -it endee /bin/sh
# Now you're inside the container!

# List files
ls /data

# Exit the container
exit

# Run a single command without entering shell
docker exec endee ls /data
```

---

## **5. Docker Workflow for This Project**

### **First Time Setup**

```bash
# 1. Start Docker Desktop (Windows/Mac)
#    Or start Docker service (Linux): sudo systemctl start docker

# 2. Navigate to project
cd e:\project\assignment_rag

# 3. Start Endee
docker compose up -d

# 4. Verify it's running
curl http://localhost:8080/status

# 5. Check logs
docker logs endee

# If you see "Server ready!" → Success! ✅
```

### **Daily Workflow**

```bash
# Morning: Start Endee
docker compose up -d

# Work on your RAG project...
# Streamlit connects to http://localhost:8080
# Endee stores vectors in endee-data volume

# Evening: Stop Endee (optional)
docker compose stop
# Data persists in volume!

# Next day: Start again
docker compose start
# All 4,772 vectors still there! ✅
```

### **When You Restart Your Computer**

```bash
# If restart: always in docker-compose.yml
# → Endee starts automatically ✅

# If not auto-starting:
docker compose up -d
```

---

## **6. Troubleshooting Docker**

### **Issue: "Docker is not running"**

```bash
# Windows/Mac: Open Docker Desktop
# Click the Docker icon in system tray

# Linux: Start Docker service
sudo systemctl start docker

# Verify
docker --version
# Should show: Docker version XX.XX.X
```

### **Issue: "Port 8080 already in use"**

```bash
# Find what's using port 8080
netstat -ano | findstr :8080  # Windows
lsof -i :8080                 # Linux/macOS

# Solution 1: Stop the other program
# Solution 2: Use different port
# Edit docker-compose.yml:
ports:
  - "8081:8080"  # Use 8081 on your PC
```

### **Issue: "Cannot connect to Docker daemon"**

```bash
# Windows/Mac: Docker Desktop not running
# → Start Docker Desktop

# Linux: Docker service stopped
sudo systemctl start docker
sudo systemctl enable docker  # Auto-start on boot
```

### **Issue: "Container keeps restarting"**

```bash
# Check logs for errors
docker logs endee

# Common causes:
# 1. Port already in use
# 2. Volume permissions issue
# 3. Corrupted image

# Solution: Remove and recreate
docker compose down
docker compose up -d
```

### **Issue: "Data disappeared after restart"**

```bash
# Did you use "docker compose down -v"?
# -v deletes volumes! ❌

# Always use:
docker compose stop  # Keeps data ✅
# OR
docker compose down  # Removes container but keeps volume ✅

# Check if volume exists
docker volume ls | grep endee-data
# If no output → Volume deleted, data lost 😞
```

---

## **7. Docker vs Virtual Machines**

### **What's the Difference?**

```
Virtual Machine (VM):
┌────────────────────────────┐
│     Application            │
│     Guest OS (Full Linux)  │ ← Entire OS!
│     Hypervisor             │
│     Host OS                │
└────────────────────────────┘
Size: 5-10 GB
Boot time: 30-60 seconds
Resources: Heavy

Docker Container:
┌────────────────────────────┐
│     Application            │
│     Container Runtime      │ ← Shares host OS!
│     Host OS                │
└────────────────────────────┘
Size: 50-500 MB
Boot time: 1-2 seconds
Resources: Light
```

**Key Difference:**
- VM = Full computer inside your computer
- Container = Isolated process on your computer

---

## **8. Why We Use Docker for Endee**

### **Benefits**

```
✅ Easy Setup
   Without Docker: "Install these 15 dependencies..."
   With Docker: "docker compose up -d" ✅

✅ Consistency
   Works same on Windows, Mac, Linux
   No "works on my machine" problems

✅ Isolation
   Endee runs in its own environment
   Doesn't conflict with other software

✅ Easy Cleanup
   Don't need it? docker compose down
   All cleaned up instantly

✅ Version Control
   image: endeeio/endee-server:latest
   Easy to upgrade or rollback
```

### **Without Docker (Manual Setup)**

```
❌ Install Python dependencies
❌ Install system libraries
❌ Configure environment variables
❌ Set up paths
❌ Troubleshoot conflicts
❌ Different steps for Windows/Mac/Linux

⏱️ Time: 30-60 minutes
😰 Frustration: High
```

### **With Docker**

```
✅ docker compose up -d

⏱️ Time: 2 minutes
😊 Frustration: Zero
```

---

## **9. Docker Cheat Sheet**

### **Container Management**

```bash
docker compose up -d        # Start (create if needed)
docker compose start        # Start (must exist)
docker compose stop         # Stop (keep container)
docker compose restart      # Restart
docker compose down         # Stop and remove
docker compose down -v      # ⚠️ Stop, remove, DELETE volumes
docker ps                   # List running
docker ps -a                # List all
docker logs endee           # View logs
docker logs endee -f        # Follow logs
docker exec -it endee sh    # Enter container shell
```

### **Image Management**

```bash
docker pull endeeio/endee-server:latest  # Download image
docker images                             # List images
docker rmi endeeio/endee-server:latest   # Remove image
docker image prune                        # Remove unused images
```

### **Volume Management**

```bash
docker volume ls                # List volumes
docker volume inspect endee-data # Inspect volume
docker volume rm endee-data     # ⚠️ Delete volume (data lost!)
docker volume prune             # Remove unused volumes
```

### **System Cleanup**

```bash
docker system df              # Show disk usage
docker system prune           # Remove unused data
docker system prune -a        # Remove everything unused
docker system prune -a --volumes  # ⚠️ Including volumes!
```

---

## **10. Real-World Examples**

### **Example 1: Fresh Start**

```bash
# Scenario: New computer, want to run Endee

# 1. Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop

# 2. Clone your project
git clone <your-repo>
cd assignment_rag

# 3. Start Endee
docker compose up -d

# 4. Verify
curl http://localhost:8080/status

# Done! ✅ Endee running with zero manual setup
```

### **Example 2: Upgrade Endee**

```bash
# Scenario: New Endee version released

# 1. Stop current version
docker compose down

# 2. Pull new image
docker pull endeeio/endee-server:latest

# 3. Start with new version
docker compose up -d

# Note: Data in volume (endee-data) preserved! ✅
```

### **Example 3: Move to Another Computer**

```bash
# Old Computer:
# 1. Backup volume
docker run --rm -v endee-data:/data -v $(pwd):/backup ubuntu tar czf /backup/endee-backup.tar.gz /data

# 2. Copy endee-backup.tar.gz to new computer

# New Computer:
# 1. Extract backup
docker run --rm -v endee-data:/data -v $(pwd):/backup ubuntu tar xzf /backup/endee-backup.tar.gz -C /

# 2. Start Endee
docker compose up -d

# All 4,772 vectors transferred! ✅
```

---

## **11. Common Questions**

### **Q: Is Docker free?**
**A:** Yes! Docker Desktop is free for personal use and small businesses.

### **Q: Does Docker slow down my computer?**
**A:** Slightly, but Endee container uses minimal resources (< 500 MB RAM).

### **Q: Can I use Docker without Docker Desktop?**
**A:** 
- Windows: Need Docker Desktop or WSL2 with Docker Engine
- Mac: Need Docker Desktop
- Linux: Can use Docker Engine directly (no Desktop needed)

### **Q: What if I delete docker-compose.yml?**
**A:** Just recreate it! Or download from GitHub. The important data is in the volume, not the compose file.

### **Q: Can I run multiple Endee instances?**
**A:** Yes! Use different ports and volume names:
```yaml
services:
  endee1:
    ports: ["8080:8080"]
    volumes: ["endee-data-1:/data"]
  endee2:
    ports: ["8081:8080"]
    volumes: ["endee-data-2:/data"]
```

### **Q: How do I completely uninstall everything?**
```bash
# Stop and remove
docker compose down -v

# Remove image
docker rmi endeeio/endee-server:latest

# Uninstall Docker Desktop (if desired)
# Windows: Control Panel → Uninstall
# Mac: Drag Docker.app to Trash
```

---

## **12. Next Steps**

Now that you understand Docker:

1. ✅ Start Endee: `docker compose up -d`
2. ✅ Verify: `curl http://localhost:8080/status`
3. ✅ Run your RAG app: `streamlit run app.py`
4. ✅ Endee handles all vector storage automatically!

**No need to understand Docker internals** - just these basic commands get you 99% of the way.

---

## **Quick Reference Card**

```bash
# Daily Use
docker compose up -d      # Start Endee
docker compose stop       # Stop (keeps data)
docker logs endee        # Check what's happening

# Troubleshooting  
docker ps                # Is it running?
docker compose restart   # Fix minor issues
docker logs endee -f     # Watch real-time logs

# Data Management
docker volume ls         # Check volumes exist
docker compose down      # Stop but keep volumes ✅
docker compose down -v   # ⚠️ DELETES volumes!

# Health Check
curl http://localhost:8080/status  # Should return {"status":"ok"}
```

---

**Congratulations!** 🎉 You now understand Docker well enough to use it for this project (and beyond).

**Remember:** Docker makes things **easier**, not harder. It's okay if you don't understand every detail - focus on the commands you need!

