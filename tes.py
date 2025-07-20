import lightning as L

# Trainer จะตรวจจับและใช้ GPU โดยอัตโนมัติ หากมีให้ใช้งานใน Environment
# การระบุแบบนี้จะช่วยให้มั่นใจว่าโค้ดร้องขอใช้ GPU อย่างถูกต้อง
trainer = L.Trainer(accelerator="gpu", devices=1)
trainer.fit(model, train_loader)