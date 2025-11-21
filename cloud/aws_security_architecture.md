# AWS Security Architecture: A Foundation in IAM

**Author:** Travis Lelle ([travis@travisml.ai](mailto:travis@travisml.ai))  
**Published:** November 2025

---

## Table of Contents
1. [Introduction](#introduction)
2. [IAM Core Concepts](#iam-core-concepts)
3. [Policy Architecture](#policy-architecture)
4. [IAM Best Practices](#iam-best-practices)
5. [Security Architecture Patterns](#security-architecture-patterns)
6. [Real-World Scenarios](#real-world-scenarios)
7. [Advanced Topics](#advanced-topics)
8. [Common Pitfalls](#common-pitfalls)

## Introduction

AWS security is built on the principle of **least privilege** - granting only the permissions required to perform a task. Identity and Access Management (IAM) is the cornerstone of AWS security, controlling who can access which resources and what actions they can perform.

This guide provides a deep dive into IAM fundamentals and how they integrate into broader AWS security architecture.

## IAM Core Concepts

### The Four Pillars of IAM

1. **Users** - Individual identities for people or applications
2. **Groups** - Collections of users with shared permissions
3. **Roles** - Temporary identities that can be assumed by users, services, or applications
4. **Policies** - JSON documents defining permissions

### Understanding the Identity Hierarchy
```
Root Account (avoid using)
    ├── IAM Users (human access)
    ├── IAM Groups (permission sets)
    ├── IAM Roles (temporary credentials)
    │   ├── Service Roles (for AWS services)
    │   ├── Cross-Account Roles (for access between accounts)
    │   └── Federation Roles (for SSO/SAML)
    └── Policies (permission definitions)
```

### Users vs Roles: When to Use What

**Use IAM Users for:**
- Individual developers needing console/CLI access
- Long-term credentials for specific team members
- Service accounts (though roles are often better)

**Use IAM Roles for:**
- EC2 instances accessing AWS services
- Lambda functions, ECS tasks, and other compute
- Cross-account access
- Temporary access for contractors or tools
- Federation (SSO integration)

**Key Principle:** Roles are almost always preferable to access keys because they provide temporary credentials that automatically rotate.

## Policy Architecture

### Policy Types

1. **Identity-based policies** - Attached to users, groups, or roles
2. **Resource-based policies** - Attached to resources (S3 buckets, KMS keys, etc.)
3. **Service Control Policies (SCPs)** - Organization-level guardrails
4. **Permission Boundaries** - Maximum permissions an identity can have
5. **Session Policies** - Limit permissions when assuming a role

### Policy Evaluation Logic

AWS evaluates policies in this order:
```
1. Explicit DENY (always wins)
2. Explicit ALLOW
3. Implicit DENY (default if no allow)
```

**Critical Rule:** A single explicit deny overrides any number of allows.

### Policy Structure Deep Dive
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowEC2Read",
      "Effect": "Allow",
      "Action": [
        "ec2:Describe*",
        "ec2:Get*"
      ],
      "Resource": "*",
      "Condition": {
        "IpAddress": {
          "aws:SourceIp": "203.0.113.0/24"
        }
      }
    }
  ]
}
```

**Component Breakdown:**

- **Version**: Policy language version (always use "2012-10-17")
- **Sid**: Statement ID (optional, for documentation)
- **Effect**: "Allow" or "Deny"
- **Action**: API calls permitted/denied
- **Resource**: ARN of resources the policy applies to
- **Condition**: Optional constraints on when the policy applies

### Practical Policy Examples

#### Example 1: Data Scientist Role - S3 and SageMaker Access
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowS3DataAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::ml-training-data/*",
        "arn:aws:s3:::ml-training-data"
      ]
    },
    {
      "Sid": "AllowSageMakerTraining",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:StopTrainingJob"
      ],
      "Resource": "*"
    },
    {
      "Sid": "AllowPassRoleForSageMaker",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
      "Condition": {
        "StringEquals": {
          "iam:PassedToService": "sagemaker.amazonaws.com"
        }
      }
    }
  ]
}
```

**Key Pattern**: The `iam:PassRole` permission is crucial when allowing services to assume roles on your behalf.

#### Example 2: Least Privilege EC2 Instance Role
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowDynamoDBAccess",
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:dynamodb:us-east-1:123456789012:table/AppData"
    },
    {
      "Sid": "AllowKMSDecrypt",
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt",
        "kms:DescribeKey"
      ],
      "Resource": "arn:aws:kms:us-east-1:123456789012:key/abc123-def456"
    },
    {
      "Sid": "AllowCloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:123456789012:log-group:/aws/ec2/myapp:*"
    }
  ]
}
```

**Design Principle**: Specify exact resources where possible. Avoid `"Resource": "*"` unless genuinely necessary.

#### Example 3: Developer with MFA Enforcement
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowAllActionsWithMFA",
      "Effect": "Allow",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "Bool": {
          "aws:MultiFactorAuthPresent": "true"
        }
      }
    },
    {
      "Sid": "DenyAllWithoutMFA",
      "Effect": "Deny",
      "NotAction": [
        "iam:CreateVirtualMFADevice",
        "iam:EnableMFADevice",
        "iam:GetUser",
        "iam:ListMFADevices",
        "iam:ListVirtualMFADevices",
        "iam:ResyncMFADevice",
        "sts:GetSessionToken"
      ],
      "Resource": "*",
      "Condition": {
        "BoolIfExists": {
          "aws:MultiFactorAuthPresent": "false"
        }
      }
    }
  ]
}
```

**Security Pattern**: Force MFA usage by denying all actions except MFA setup when MFA isn't present.

## IAM Best Practices

### 1. Root Account Protection

**Never use root account for daily operations:**
```bash
# Set up root account properly:
1. Enable MFA on root account (hardware token recommended)
2. Delete root access keys (if they exist)
3. Create individual IAM users for all admins
4. Lock root credentials in secure location
5. Set up CloudWatch alarm for root account usage
```

**CloudWatch Alarm for Root Usage:**
```json
{
  "filterName": "RootAccountUsage",
  "filterPattern": "{ $.userIdentity.type = \"Root\" && $.userIdentity.invokedBy NOT EXISTS && $.eventType != \"AwsServiceEvent\" }",
  "metricTransformations": [
    {
      "metricName": "RootAccountUsageCount",
      "metricNamespace": "Security",
      "metricValue": "1"
    }
  ]
}
```

### 2. Enable MFA Everywhere

- **Root account**: Hardware MFA token
- **Admin users**: Virtual or hardware MFA
- **Regular users**: Virtual MFA minimum
- **Programmatic access**: Use roles instead, or short-lived credentials with session MFA

### 3. Use IAM Roles Over Access Keys

**Bad Practice:**
```bash
# Embedding access keys in application code
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

**Good Practice:**
```bash
# EC2 instance with IAM role
aws ec2 run-instances \
  --image-id ami-12345678 \
  --instance-type t3.micro \
  --iam-instance-profile MyAppRole
```

The instance automatically gets temporary credentials via instance metadata service.

### 4. Implement Least Privilege Gradually

**Start restrictive, expand as needed:**
```
1. Start with minimal permissions
2. Monitor CloudTrail for AccessDenied errors
3. Add only the permissions needed
4. Use Access Analyzer to refine policies
5. Review permissions quarterly
```

### 5. Use Permission Boundaries

Permission boundaries set the maximum permissions an entity can have:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:*",
        "dynamodb:*",
        "lambda:*"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Deny",
      "Action": [
        "iam:*",
        "organizations:*",
        "account:*"
      ],
      "Resource": "*"
    }
  ]
}
```

**Use Case**: Allow developers to create roles for their applications, but prevent them from escalating privileges.

### 6. Credential Rotation Strategy

**Manual Access Keys:**
```bash
# Check key age
aws iam get-credential-report

# Rotate keys every 90 days
aws iam create-access-key --user-name developer1
aws iam delete-access-key --user-name developer1 --access-key-id OLD_KEY
```

**Automated with Secrets Manager:**
```python
import boto3

secrets = boto3.client('secretsmanager')

# Store initial credentials
secrets.create_secret(
    Name='prod/database/credentials',
    SecretString='{"username":"admin","password":"initial_pass"}',
    Description='Production database credentials'
)

# Enable automatic rotation
secrets.rotate_secret(
    SecretId='prod/database/credentials',
    RotationLambdaARN='arn:aws:lambda:region:account:function:RotateDBCreds',
    RotationRules={'AutomaticallyAfterDays': 30}
)
```

### 7. Use Service Control Policies (SCPs)

SCPs provide guardrails at the organization level:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyRegionsOutsideUS",
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "StringNotEquals": {
          "aws:RequestedRegion": [
            "us-east-1",
            "us-west-2"
          ]
        }
      }
    },
    {
      "Sid": "DenyLeavingOrganization",
      "Effect": "Deny",
      "Action": "organizations:LeaveOrganization",
      "Resource": "*"
    }
  ]
}
```

## Security Architecture Patterns

### Pattern 1: Multi-Account Strategy

**Design:**
```
Organization Root
├── Security Account (CloudTrail, GuardDuty, Security Hub)
├── Log Archive Account (centralized logging)
├── Shared Services (networking, directory)
├── Production Account
├── Development Account
└── Sandbox Accounts (isolated experimentation)
```

**Benefits:**
- Blast radius containment
- Separate billing/budget controls
- Clear environment boundaries
- Simplified compliance auditing

**Cross-Account Role Example:**
```json
// In Production Account - Trust policy for role
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::111111111111:root"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "sts:ExternalId": "unique-external-id-12345"
        }
      }
    }
  ]
}
```
```bash
# Assuming role from Development Account
aws sts assume-role \
  --role-arn arn:aws:iam::222222222222:role/ProdReadOnly \
  --role-session-name dev-session \
  --external-id unique-external-id-12345
```

### Pattern 2: Data Tier Security

**Architecture Layers:**
```
┌─────────────────────────────────────┐
│   Application Layer                 │
│   - EC2/ECS with app-specific role │
│   - Can only access own data       │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│   Data Layer                        │
│   - DynamoDB with table policies    │
│   - S3 with bucket policies         │
│   - RDS with IAM auth              │
└─────────────────────────────────────┘
```

**DynamoDB with Fine-Grained Access:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/UserData",
      "Condition": {
        "ForAllValues:StringEquals": {
          "dynamodb:LeadingKeys": [
            "${aws:userid}"
          ]
        }
      }
    }
  ]
}
```

This allows users to only access items where the partition key matches their user ID.

### Pattern 3: Network Security with IAM

**VPC Endpoint Policies:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowS3ReadFromPrivateSubnet",
      "Effect": "Allow",
      "Principal": "*",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-secure-bucket",
        "arn:aws:s3:::my-secure-bucket/*"
      ],
      "Condition": {
        "StringEquals": {
          "aws:SourceVpce": "vpce-12345678"
        }
      }
    }
  ]
}
```

**Combined with S3 Bucket Policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyNonVPCAccess",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::my-secure-bucket",
        "arn:aws:s3:::my-secure-bucket/*"
      ],
      "Condition": {
        "StringNotEquals": {
          "aws:SourceVpce": "vpce-12345678"
        }
      }
    }
  ]
}
```

**Result**: Bucket can only be accessed through the VPC endpoint, never from internet.

### Pattern 4: Lambda Security

**Lambda Execution Role:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem"
      ],
      "Resource": "arn:aws:dynamodb:us-east-1:123456789012:table/Events"
    },
    {
      "Effect": "Allow",
      "Action": "kms:Decrypt",
      "Resource": "arn:aws:kms:us-east-1:123456789012:key/*",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "dynamodb.us-east-1.amazonaws.com"
        }
      }
    }
  ]
}
```

**Resource Policy (who can invoke):**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowAPIGatewayInvoke",
      "Effect": "Allow",
      "Principal": {
        "Service": "apigateway.amazonaws.com"
      },
      "Action": "lambda:InvokeFunction",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:MyFunction",
      "Condition": {
        "ArnLike": {
          "AWS:SourceArn": "arn:aws:execute-api:us-east-1:123456789012:abc123/*"
        }
      }
    }
  ]
}
```

## Real-World Scenarios

### Scenario 1: ML Pipeline Security

**Requirements:**
- Data scientists need to train models
- Training data is sensitive
- Models should be versioned in S3
- Prevent accidental data deletion

**Architecture:**
```
Data Scientists (IAM Users in "DataScience" Group)
    ↓ (assume role)
SageMaker Execution Role
    ↓ (reads from)
S3 Training Data Bucket (with versioning + MFA delete)
    ↓ (writes models to)
S3 Model Artifact Bucket (with lifecycle policies)
```

**Data Scientist Policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowSageMakerNotebookCreation",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateNotebookInstance",
        "sagemaker:DescribeNotebookInstance",
        "sagemaker:StartNotebookInstance",
        "sagemaker:StopNotebookInstance"
      ],
      "Resource": "arn:aws:sagemaker:*:*:notebook-instance/ds-*",
      "Condition": {
        "StringEquals": {
          "sagemaker:RootAccess": "Disabled"
        }
      }
    },
    {
      "Sid": "AllowPassRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::123456789012:role/SageMakerMLRole",
      "Condition": {
        "StringEquals": {
          "iam:PassedToService": "sagemaker.amazonaws.com"
        }
      }
    }
  ]
}
```

**S3 Bucket Policy (Training Data):**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowSageMakerRead",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:role/SageMakerMLRole"
      },
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::ml-training-data",
        "arn:aws:s3:::ml-training-data/*"
      ]
    },
    {
      "Sid": "DenyDeletion",
      "Effect": "Deny",
      "Principal": "*",
      "Action": [
        "s3:DeleteObject",
        "s3:DeleteObjectVersion"
      ],
      "Resource": "arn:aws:s3:::ml-training-data/*"
    }
  ]
}
```

### Scenario 2: CI/CD Pipeline Security

**Requirements:**
- GitHub Actions needs to deploy to AWS
- Should only deploy to specific environments
- No long-lived credentials

**Solution: OIDC Federation**
```json
// Trust policy for GitHub Actions role
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:myorg/myrepo:ref:refs/heads/main"
        }
      }
    }
  ]
}
```

**Deployment Role Policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowECSDeploy",
      "Effect": "Allow",
      "Action": [
        "ecs:UpdateService",
        "ecs:DescribeServices",
        "ecs:RegisterTaskDefinition"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:ResourceTag/Environment": "production"
        }
      }
    },
    {
      "Sid": "AllowECRPush",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "arn:aws:ecr:us-east-1:123456789012:repository/myapp"
    }
  ]
}
```

**GitHub Actions Workflow:**
```yaml
# .github/workflows/deploy.yml
jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    steps:
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsRole
          aws-region: us-east-1
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster production \
            --service myapp \
            --force-new-deployment
```

**No credentials stored in GitHub!**

### Scenario 3: Third-Party Integration

**Requirements:**
- Vendor needs read-only access to specific S3 buckets
- Access should be time-limited
- Audit all vendor actions

**Solution: External ID + Session Duration**
```json
// Trust policy with External ID
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::vendor-account-id:root"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "sts:ExternalId": "unique-external-id-abc123"
        },
        "IpAddress": {
          "aws:SourceIp": [
            "203.0.113.0/24"
          ]
        }
      }
    }
  ]
}
```

**Permissions:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::vendor-accessible-data",
        "arn:aws:s3:::vendor-accessible-data/*"
      ]
    }
  ]
}
```

**Role Configuration:**
```bash
# Set maximum session duration to 1 hour
aws iam update-role \
  --role-name VendorReadOnlyRole \
  --max-session-duration 3600
```

**CloudTrail Alert for Vendor Activity:**
```json
{
  "filterPattern": "{ $.userIdentity.type = \"AssumedRole\" && $.userIdentity.principalId = \"*:VendorSession*\" }",
  "metricName": "VendorAPIActivity",
  "metricNamespace": "Security",
  "metricValue": "1"
}
```

## Advanced Topics

### 1. IAM Policy Simulator

Test policies before deployment:
```bash
# Test if role can perform action
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::123456789012:role/MyRole \
  --action-names s3:GetObject \
  --resource-arns arn:aws:s3:::mybucket/*

# Test custom policy
aws iam simulate-custom-policy \
  --policy-input-list file://policy.json \
  --action-names ec2:RunInstances \
  --resource-arns "*"
```

### 2. Access Analyzer

Identify resources shared with external entities:
```bash
# Create analyzer
aws accessanalyzer create-analyzer \
  --analyzer-name MyOrgAnalyzer \
  --type ORGANIZATION

# List findings
aws accessanalyzer list-findings \
  --analyzer-arn arn:aws:access-analyzer:us-east-1:123456789012:analyzer/MyOrgAnalyzer
```

**Common findings:**
- S3 buckets with public access
- KMS keys shared with other accounts
- IAM roles with overly permissive trust policies
- Lambda functions with resource-based policies allowing external invocation

### 3. Attribute-Based Access Control (ABAC)

Use tags for dynamic access control:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:*",
      "Resource": "arn:aws:s3:::*",
      "Condition": {
        "StringEquals": {
          "s3:ExistingObjectTag/Owner": "${aws:username}",
          "aws:PrincipalTag/Department": "${aws:ResourceTag/Department}"
        }
      }
    }
  ]
}
```

**Benefits:**
- Scales better than explicit resource ARNs
- Reduces policy count
- Self-service access based on attributes

**Tagging Strategy:**
```bash
# Tag user
aws iam tag-user \
  --user-name john.doe \
  --tags Key=Department,Value=Engineering Key=Project,Value=MLPlatform

# Tag S3 object
aws s3api put-object-tagging \
  --bucket my-bucket \
  --key data/file.csv \
  --tagging '{"TagSet":[{"Key":"Department","Value":"Engineering"},{"Key":"Owner","Value":"john.doe"}]}'
```

### 4. IAM Access Advisor

Analyze when services were last accessed:
```bash
# Generate report
JOB_ID=$(aws iam generate-service-last-accessed-details \
  --arn arn:aws:iam::123456789012:role/MyRole \
  --query 'JobId' \
  --output text)

# Get report
aws iam get-service-last-accessed-details \
  --job-id $JOB_ID
```

**Use this to:**
- Identify unused permissions
- Right-size policies
- Remove services that haven't been accessed in 90+ days

### 5. CloudTrail for IAM Auditing

**Essential Events to Monitor:**
```json
{
  "eventName": [
    "CreateUser",
    "DeleteUser",
    "CreateAccessKey",
    "DeleteAccessKey",
    "AttachUserPolicy",
    "AttachRolePolicy",
    "PutUserPolicy",
    "PutRolePolicy",
    "CreateRole",
    "DeleteRole",
    "UpdateAssumeRolePolicy"
  ]
}
```

**CloudWatch Insights Query:**
```sql
fields @timestamp, userIdentity.principalId, eventName, errorCode
| filter eventName like /Create|Delete|Attach|Put/
| filter eventSource = "iam.amazonaws.com"
| sort @timestamp desc
| limit 100
```

**S3 Select for CloudTrail Analysis:**
```python
import boto3

s3 = boto3.client('s3')

response = s3.select_object_content(
    Bucket='cloudtrail-logs-bucket',
    Key='AWSLogs/123456789012/CloudTrail/us-east-1/2024/11/19/log.json.gz',
    Expression="""
        SELECT * FROM s3object[*].Records[*] s 
        WHERE s.eventName IN ('CreateUser', 'DeleteUser', 'AttachUserPolicy')
    """,
    ExpressionType='SQL',
    InputSerialization={'JSON': {'Type': 'LINES'}, 'CompressionType': 'GZIP'},
    OutputSerialization={'JSON': {}}
)

for event in response['Payload']:
    if 'Records' in event:
        print(event['Records']['Payload'].decode())
```

## Common Pitfalls

### 1. Overly Broad Wildcards

**Bad:**
```json
{
  "Effect": "Allow",
  "Action": "*",
  "Resource": "*"
}
```

**Better:**
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:Get*",
    "s3:List*"
  ],
  "Resource": [
    "arn:aws:s3:::specific-bucket",
    "arn:aws:s3:::specific-bucket/*"
  ]
}
```

### 2. Embedded Credentials

**Never do this:**
```python
# WRONG - credentials in code
client = boto3.client(
    's3',
    aws_access_key_id='AKIAIOSFODNN7EXAMPLE',
    aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
)
```

**Instead:**
```python
# RIGHT - use IAM role or credential provider
client = boto3.client('s3')  # Automatically uses IAM role
```

### 3. Forgetting Resource-Based Policies

Remember: You need BOTH identity AND resource policies to allow in some cases:
```
EC2 Instance Role + S3 Bucket Policy = Access Granted
                 (AND operation)
```

### 4. Principal Confusion in Trust Policies

**Vulnerable:**
```json
{
  "Principal": {
    "AWS": "*"
  }
}
```

This allows ANY AWS account to assume the role!

**Fixed:**
```json
{
  "Principal": {
    "AWS": "arn:aws:iam::123456789012:root"
  },
  "Condition": {
    "StringEquals": {
      "sts:ExternalId": "unique-id"
    }
  }
}
```

### 5. Not Using Conditions

Conditions add critical constraints:
```json
{
  "Effect": "Allow",
  "Action": "s3:*",
  "Resource": "*",
  "Condition": {
    "Bool": {
      "aws:SecureTransport": "true"
    },
    "StringEquals": {
      "s3:x-amz-server-side-encryption": "AES256"
    }
  }
}
```

This enforces HTTPS and encryption.

### 6. Ignoring Session Policies

When assuming a role, you can further restrict permissions:
```bash
aws sts assume-role \
  --role-arn arn:aws:iam::123456789012:role/AdminRole \
  --role-session-name temp-session \
  --policy '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::specific-bucket/*"
    }]
  }'
```

Even though AdminRole has full permissions, this session is limited to S3 read.

### 7. Not Monitoring Privilege Escalation Paths

**Dangerous permission:**
```json
{
  "Effect": "Allow",
  "Action": "iam:PutUserPolicy",
  "Resource": "*"
}
```

This allows creating arbitrary policies, effectively granting admin access.

**Watch for:**
- `iam:PutUserPolicy`
- `iam:AttachUserPolicy` with `*`
- `iam:CreateAccessKey` on other users
- `iam:UpdateAssumeRolePolicy`
- `sts:AssumeRole` without restrictions

## Security Checklist

### Initial Setup
- [ ] Enable MFA on root account
- [ ] Delete root access keys
- [ ] Create admin IAM user with MFA
- [ ] Enable CloudTrail in all regions
- [ ] Create billing alarm
- [ ] Set up AWS Organizations (multi-account)

### Ongoing Maintenance
- [ ] Review IAM users quarterly - remove unused
- [ ] Rotate access keys every 90 days
- [ ] Review IAM policies for overly broad permissions
- [ ] Check Access Analyzer findings weekly
- [ ] Run IAM Credential Report monthly
- [ ] Audit CloudTrail logs for suspicious activity
- [ ] Review and update SCPs

### Policy Development
- [ ] Start with least privilege
- [ ] Use IAM Policy Simulator to test
- [ ] Document purpose of each policy
- [ ] Specify resources explicitly (avoid `*`)
- [ ] Add conditions where applicable
- [ ] Use permission boundaries for delegation
- [ ] Version control all custom policies

### Incident Response
- [ ] Document role assumption chains
- [ ] Have break-glass access procedure
- [ ] Know how to revoke all sessions for a role
- [ ] Monitor for `ConsoleLogin` events from unusual locations
- [ ] Set up GuardDuty for automated threat detection

## Conclusion

IAM is the foundation of AWS security. Key takeaways:

1. **Least privilege is a journey, not a destination** - Start restrictive, iterate based on actual usage
2. **Roles over keys** - Use temporary credentials whenever possible
3. **Defense in depth** - Layer identity policies, resource policies, SCPs, and network controls
4. **Audit continuously** - CloudTrail, Access Analyzer, and Credential Reports are your friends
5. **Automate security** - Use tools like CloudFormation/Terraform to enforce standards

Security is not a one-time setup but an ongoing practice. Build security into your architecture from day one, and review regularly as your environment evolves.

---

*Author: [Travis Lelle]*  
*Last Updated: November 2025*  
*Feedback: [travis@travisml.ai]*